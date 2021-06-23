import argparse

import sys
from os import listdir

import torch
import torchvision

from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
from torch.utils.data import Dataset
import construct_dataset
import transforms as T
import utils
import optuna
import matplotlib.pyplot as plt
import numpy as np
import random


class TrainOREvaluate(object):

    def __init__(self):
        # self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.dataset = args.dataset
        # self.batch_size = args.batch_size
        self.train_size = args.train_size
        self.best_val_loss = float('Inf')

    def suggest_hyperparameters(trial):
        # Learning rate on a logarithmic scale
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        # Batch size  in the range from 2 to 6 with step size 1
        batch_size = int(trial.suggest_float("batch_size", 1, 4, step=1))
        optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam"])

        return lr, batch_size, optimizer_name

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train(self, trial):
        print("Training day and night")
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)

        g = torch.Generator()
        g.manual_seed(0)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        lr, batch_size, optimizer_name = TrainOREvaluate.suggest_hyperparameters(trial)

        print('Training on :', device)

        if self.dataset == 'normal':
            annotation_list = torch.load(
                '/../../data/processed/train/annotation_list.pt')
            images_list = torch.load('/../../data/processed/train/images_list.pt')
        elif self.dataset == 'augmented':
            annotation_list = torch.load(
                '/../../data/processed/train/annotation_list_augmented.pt')
            images_list = torch.load(
                '/../../data/processed/train/images_list_augmented.pt')

        train_dataset = construct_dataset.constructDataset(annotation_list, images_list, transform=None)

        train_len = int(self.train_size * len(train_dataset))
        valid_len = len(train_dataset) - train_len

        train_set, validation_set = torch.utils.data.random_split(train_dataset, lengths=[train_len, valid_len])

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn, worker_init_fn=TrainOREvaluate.seed_worker,
            generator=g)  # collate_fn allows you to have images (tensors) of different sizes

        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn, worker_init_fn=TrainOREvaluate.seed_worker,
            generator=g)  # collate_fn allows you to have images (tensors) of different sizes

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (sheep) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained final layer classification and box regression layers with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        torch.save(model.state_dict(), '/../../models/sheep_vanilla.pth')

        model.to(device)

        # Pick an optimizer based on Optuna's parameter suggestion
        params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr,
                                        momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        metric_collector = []

        for epoch in range(self.num_epochs):
            # train for one epoch, printing every 5 iterations
            metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5)
            metric_collector.append(metric_logger)
            # update the learning rate
            lr_scheduler.step()
            # Evaluate with validation dataset
            evaluation_result = evaluate(model, validation_loader, device=device)
            validation_AP_accuracy = evaluation_result.coco_eval.get('bbox').stats[0]

            trial.report(validation_AP_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # save checkpoint

            torch.save(model.state_dict(),
                       '/../../models/sheep_train_' + self.dataset + '.pth')

        # Creating the test set and testing
        annotation_list = torch.load(
            '/../../data/processed/test/annotations_test.pt')
        images_list = torch.load('/../../data/processed/test/images_test.pt')

        test_dataset = construct_dataset.constructDataset(annotation_list, images_list, transform=None)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn, worker_init_fn=TrainOREvaluate.seed_worker,
            generator=g)  # collate_fn allows you to have images (tensors) of different sizes

        evaluation_result = evaluate(model, test_loader, device=device)

        final_accuracy = evaluation_result.coco_eval.get('bbox').stats[0]

        if final_accuracy <= self.best_val_loss:
            self.best_val_loss = final_accuracy
            print(self.best_val_loss)

        return final_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-lr',
    #                     default=0.005,
    #                     type=float)
    parser.add_argument('-num_epochs',
                        default=3,
                        type=int)
    parser.add_argument('-dataset',
                        default='augmented',
                        type=str)
    #  parser.add_argument('-batch_size',
    #                      default=4,
    #                      type=int)
    parser.add_argument('-train_size',
                        default=0.9,
                        type=float)
    args = parser.parse_args()

    trainObj = TrainOREvaluate()
    #    final_accuracy = trainObj.train(lr)

    study = optuna.create_study(study_name="Final-project-optuna", direction="maximize",
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=5, n_warmup_steps=1
                                ))
    study.optimize(trainObj.train, n_trials=20)

    # Initialize the best_val_loss value
    # mean_AP_accuracy = best_val_loss = float('Inf')

    # if mean_AP_accuracy <= best_val_loss:
    #     best_val_loss = mean_AP_accuracy

    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    lr = 0
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig_history = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig('fig_history.png')
    fig_param_importances = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig('fig_param_importances.png')
    plot_edf = optuna.visualization.matplotlib.plot_edf(study)
    plt.savefig('plot_edf.png')
    intermediate_values = optuna.visualization.matplotlib.plot_intermediate_values(study)
    plt.savefig('intermediate_values.png')
    parallel_coordinate = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig('parallel_coordinate.png')




