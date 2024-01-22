#!/usr/bin/env python

import torch
import torchvision
# from torchvision.models import ResNet50_Weights
import swanlab
from torch.utils.data import DataLoader
from load_datasets import DatasetLoader
import os
import csv
from datetime import datetime

# 训练开始时间
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 新增代码：创建CSV文件用于保存训练结果
csv_filename = f"output/training_results_{current_time}.csv"
csv_columns = ["Epoch", "Train Loss", "Train Accuracy", "Val Accuracy"]

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()


# 定义训练函数
def train(model, device, train_dataloader, optimizer, criterion, epoch):
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
                epoch, num_epochs, iter + 1, len(TrainDataLoader), loss.item()
            )
        )

    # 计算平均损失和训练集准确率
    average_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_train / total_train

    swanlab.log({"train_loss": average_loss, "train_acc": train_accuracy})

    return {
        "Epoch": epoch,
        "Train Loss": average_loss,
        "Train Accuracy": train_accuracy,
    }


# 定义验证函数
def val(model, device, val_dataloader):
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

    swanlab.log({"val_acc": accuracy})

    return {"Val Accuracy": accuracy}


if __name__ == "__main__":
    num_epochs = 70
    lr = 1e-4
    batch_size = 50
    num_classes = 2

    # 设置device
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

    # 初始化swanlab
    swanlab.init(
        experiment_name="ResNet50",
        description="Train ResNet50 for stained and non-stained classification.",
        config={
            "model": "resnet50",
            "optim": "Adam",
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_class": num_classes,
            "device": device,
        },
    )

    TrainDataset = DatasetLoader("datasets/train.csv")
    ValDataset = DatasetLoader("datasets/val.csv")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)

    # 载入ResNet50模型
    # model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torchvision.models.resnet50(weights=None)

    # 将全连接层替换为2分类
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    model.to(torch.device(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for epoch in range(1, num_epochs + 1):
        train_accu = train(model, device, TrainDataLoader, optimizer, criterion, epoch)
        val_accu = val(model, device, ValDataLoader)

        # 合并
        train_accu.update(val_accu)

        # 保存到CSV文件
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writerow(train_accu)

    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    filename = f"checkpoint/latest_checkpoint_{current_time}.pth"

    # 保存权重文件
    torch.save(model.state_dict(), filename)

    print("Training complete")
