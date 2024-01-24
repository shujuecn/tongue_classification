#!/usr/bin/env python

import csv
import os
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from load_datasets import DatasetLoader


# 定义训练函数
def train(model, device, train_dataloader, optimizer, criterion, epoch, num_epochs):
    model.train()

    total_loss = 0
    correct_train = 0
    total_train = 0

    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        print(
            "Epoch [{}/{}]\tIteration [{}/{}]\tLoss: {:.4f}".format(
                epoch, num_epochs, iter + 1, len(train_dataloader), loss.item()
            )
        )

    # 计算平均损失和训练集准确率
    average_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_train / total_train

    writer.add_scalar("train_loss", scalar_value=average_loss, global_step=epoch)
    writer.add_scalar("train_accu", scalar_value=train_accuracy, global_step=epoch)

    return {
        "Epoch": epoch,
        "Train Loss": average_loss,
        "Train Accuracy": train_accuracy,
    }


# 定义验证函数
def val(model, device, val_dataloader, epoch):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("Accuracy: {:.2f}".format(accuracy))

    writer.add_scalar("val_accu", scalar_value=accuracy, global_step=epoch)

    return {"Val Accuracy": accuracy}


def get_device():
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)


def run(num_epochs, lr, batch_size, num_classes):
    TrainDataset = DatasetLoader("datasets/train.csv")
    ValDataset = DatasetLoader("datasets/val.csv")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)

    # 载入ResNet50模型
    model = torchvision.models.resnet50(weights=None)

    # 将全连接层替换为2分类
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for epoch in range(1, num_epochs + 1):
        train_accu = train(
            model, device, TrainDataLoader, optimizer, criterion, epoch, num_epochs
        )
        val_accu = val(model, device, ValDataLoader, epoch)

        # 合并
        train_accu.update(val_accu)

        # 保存到CSV文件
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writerow(train_accu)

    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    model_name = f"checkpoint/{current_time}_{file_name}.pth"

    # 保存权重文件
    torch.save(model.state_dict(), model_name)

    print("Training complete")


if __name__ == "__main__":
    # 模型参数
    num_epochs = 70
    lr = 1e-5
    batch_size = 96
    num_classes = 2

    device = get_device()
    current_time = datetime.now().strftime("%m%d%H%M")
    file_name = f"NE{num_epochs}_BS{batch_size}_LR{lr}"

    # 保存训练结果
    csv_filename = f"output/{current_time}_{file_name}.csv"
    csv_columns = ["Epoch", "Train Loss", "Train Accuracy", "Val Accuracy"]

    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

    # 开启日志，启动训练
    writer = SummaryWriter(comment=f"_{file_name}")
    run(num_epochs, lr, batch_size, num_classes)
    writer.close()
