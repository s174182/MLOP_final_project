import torch
from torch.utils.data import Dataset


class costructDataset(Dataset):

    def __init__(self, annotation_list, images_list, transform=None):
        self.load_box = annotation_list
        self.load_image = images_list
        self.transforms = transform

    def __getitem__(self, idx, transform=None):
        boxes = self.load_box[idx]  # return list of [xmin, ymin, xmax, ymax]
        img = self.load_image[idx]  # return an image
        self.transform = transform

        if len(list(boxes.shape)) > 1:

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            num_box = boxes.shape[0]

        else:
            # negative example, ref: https://github.com/pytorch/vision/issues/2144

            boxes = boxes.unsqueeze_(0)
            num_box = 1

        labels = torch.ones((num_box,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        issheep = torch.zeros((num_box,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["issheep"] = issheep

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.load_image)
