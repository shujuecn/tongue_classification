import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


def getStat(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print("Compute mean and variance for training data.")
    print(len(train_data))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()

    mean.div_(len(train_data))
    std.div_(len(train_data))

    return list(mean.numpy()), list(std.numpy())


if __name__ == "__main__":
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageFolder(root="datasets/train", transform=data_transform)
    print(getStat(train_dataset))
