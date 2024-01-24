#!/usr/bin/env python

import csv
import logging
import os
from datetime import datetime
import io

import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models, transforms
import torch.nn.functional as F


from load_datasets import DatasetLoader


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


def set_logger(filename):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def load_resnet50(num_classes=2, checkpoint_path=None, evaluate=False):
    """加载ResNet50模型"""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=get_device()))

    if evaluate:
        model.eval()  # Set model to evaluation mode

    return model


def calculate_accuracy(predicted_labels, true_labels):
    """计算准确率"""
    correct_predictions = torch.sum(predicted_labels == torch.tensor(true_labels))
    total_samples = len(predicted_labels)
    accuracy = correct_predictions / total_samples
    return accuracy


def process_image(image, image_size):
    """加载图像并执行必要的转换的函数"""
    preprocessing = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # TODO
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = preprocessing(image).unsqueeze(0)
    return image


def predict(model, image):
    """预测单张图像类别并返回概率"""
    image = process_image(image, 256)  # Using the image size from training
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(
            outputs, dim=1
        ).squeeze()  # Apply softmax to get probabilities
    return probabilities


def predict_folder(model, image_folder):
    """预测指定文件夹的所有图片"""
    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        # 类别概率
        probabilities = predict(model, image)

        classes = {
            "0": "stained",
            "1": "non-stained",
        }  # Update or extend this dictionary based on your actual classes
        class_probabilities = {
            classes[str(i)]: round(float(prob), 4)
            for i, prob in enumerate(probabilities)
        }

        print(
            f"{image_file}\t{class_probabilities}\t{max(class_probabilities, key=class_probabilities.get)}"
        )


def train(model, train_dataloader, optimizer, criterion, epoch, num_epochs):
    """训练函数"""
    model.train()

    total_loss = 0
    correct_train = 0
    total_train = 0

    for index, (inputs, labels) in enumerate(train_dataloader):
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

        logger.info(
            "Epoch [{}/{}]\tIteration [{}/{}]\tLoss: {:.4f}".format(
                epoch, num_epochs, index + 1, len(train_dataloader), loss.item()
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


def val(model, val_dataloader, epoch):
    """验证函数"""
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

    logger.info("Val Accuracy: {:.4f}".format(accuracy))
    writer.add_scalar("val_accu", scalar_value=accuracy, global_step=epoch)

    return {"Val Accuracy": accuracy}


def test(checkpoint, test_data):
    """测试函数"""

    model = load_resnet50(checkpoint_path=checkpoint, evaluate=True)

    df = pd.read_csv(test_data, header=None, names=["Filepath", "TrueLabel"])
    # Create an empty tensor to store predicted labels
    predicted_labels = torch.zeros(len(df))

    # Use a loop to fill the tensor with predicted labels
    for i, (_, row) in enumerate(df.iterrows()):
        image_path = os.path.join("datasets/", row["Filepath"])
        image = Image.open(image_path)
        prediction = predict(model, image)
        predicted_label = torch.argmax(prediction).item()
        predicted_labels[i] = predicted_label

    # Calculate accuracy
    accuracy = calculate_accuracy(predicted_labels, df["TrueLabel"])
    # logger.info(f"Test Accuracy: {accuracy:.4f}")

    # return {"Test Accuracy": accuracy}
    return accuracy


def plot(csv_filename, batch_size, lr, test_accuracy):
    """绘制曲线"""

    # 读取CSV文件
    df = pd.read_csv(csv_filename)

    # 提取数据
    epochs = df["Epoch"]
    train_loss = df["Train Loss"]
    train_accuracy = df["Train Accuracy"]
    val_accuracy = df["Val Accuracy"]

    # 绘制折线图
    # plt.figure(figsize=(10, 7), dpi=300)
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color)
    ax1.plot(
        epochs, train_loss, color=color, linestyle="--", label="Train Loss", linewidth=1
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(
        epochs,
        train_accuracy,
        color=color,
        linestyle="--",
        label="Train Accuracy",
        linewidth=1,
    )
    ax2.plot(
        epochs,
        val_accuracy,
        color=color,
        linestyle="-",
        label="Val Accuracy",
        linewidth=1,
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # 设置y轴标签格式
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    # 显示图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines + lines2,
        labels + labels2,
        loc="upper left",
        bbox_to_anchor=(0.69, 1.21),
        frameon=False,
    )

    info_text = (
        f"Batch size: {int(batch_size)}\n"
        f"Learning rate: {float(lr)}\n"
        f"Average training accuracy: {train_accuracy.mean():.4f}\n"
        f"Average verification accuracy: {val_accuracy.mean():.4f}\n"
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    fig.text(
        0.12,
        1.1,
        info_text,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
        linespacing=1.6,
    )

    plt.savefig(
        f"./output/image/{csv_filename.split('/')[-1].split('.')[0]}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def main(num_epochs, lr, batch_size):
    TrainDataset = DatasetLoader("datasets/train.csv")
    ValDataset = DatasetLoader("datasets/val.csv")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)

    # 载入ResNet50模型
    model = load_resnet50()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for epoch in range(1, num_epochs + 1):
        train_accu = train(
            model, TrainDataLoader, optimizer, criterion, epoch, num_epochs
        )
        val_accu = val(model, ValDataLoader, epoch)
        train_accu.update(val_accu)

        # 保存到CSV文件
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writerow(train_accu)

    model_name = f"checkpoint/{current_time}_{file_name}.pth"

    # 保存权重文件
    torch.save(model.state_dict(), model_name)
    print(f"Training complete, model saved as '{model_name}'")

    # 测试集
    # test_accu = test(checkpoint=model_name, test_data="datasets/test.csv")

    # 绘制曲线
    # plot(test_accu)


if __name__ == "__main__":
    # 模型参数
    num_epochs = 60
    lr = 1e-5
    batch_size = 96
    num_classes = 2

    device = get_device()
    current_time = datetime.now().strftime("%m%d%H%M")
    file_name = f"NE_{num_epochs}_BS_{batch_size}_LR_{lr}"

    os.makedirs("output/csv", exist_ok=True)
    os.makedirs("output/image", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)

    # 保存训练结果
    csv_filename = f"output/csv/{current_time}_{file_name}.csv"
    csv_columns = ["Epoch", "Train Loss", "Train Accuracy", "Val Accuracy"]

    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

    logger = set_logger(f"logs/{current_time}_{file_name}.log")
    writer = SummaryWriter(comment=f"_{file_name}")

    main(num_epochs, lr, batch_size)

    writer.close()
