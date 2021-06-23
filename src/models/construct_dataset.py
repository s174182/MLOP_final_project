import torch
from torch.utils.data import Dataset


class constructDataset(Dataset):
    def __init__(self, annotation_list, images_list, transform=None):
        self.load_box = annotation_list
        self.load_image = images_list
        self.transforms = transform

    def __getitem__(self, idx, transform=None):

        boxes = self.load_box[idx].get(
            "boxes"
        )  # return list of [xmin, ymin, xmax, ymax]
        boxes = torch.as_tensor(boxes, dtype=torch.float16)
        boxes = boxes.reshape(-1, 4)  # Reshape so that it always has 2 dimensions

        img = self.load_image[idx]  # return a tensor
        img = torch.squeeze(img)  # TODO: maybe we unsqueeze later

        num_box = boxes.shape[0]

        labels = torch.ones((num_box,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_box,), dtype=torch.int8)

        target = {}
        target["boxes"] = boxes.float()
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area.float()
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.load_image)
