import csv
import functools
import os
from datetime import datetime
import logging

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from .models import get_model
from .dataset import DataHandler


class TongueClassifier:
    def __init__(
        self,
        model_name: str,
        data_dir: str,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        random_seed: int,
    ):
        self.model = get_model(model_name)
        self.device = self.get_device()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.seed = random_seed

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=0
        )

        self.summary_writer: SummaryWriter

        self.start_time = self.current_time()
        self.file_name = f"{self.start_time}_{self.model.name}_NE_{self.num_epochs}_BS_{self.batch_size}_LR_{self.lr}"

        self.setup_experiment(self.start_time)

    @staticmethod
    def current_time():
        return datetime.now().strftime("%m%d%H%M")

    def setup_experiment(self, start_time):

        os.makedirs("output/train_info", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs(f"checkpoint/{self.model.name}", exist_ok=True)
        os.makedirs(f"croped_images/split_info/{self.file_name}", exist_ok=True)

        self.logger = self.set_logger(f"logs/{self.file_name}.log")

        # 保存训练结果
        self.csv_filename = f"output/train_info/{self.file_name}.csv"
        self.csv_columns = ["Epoch", "Train Loss", "Train Accuracy", "Val Accuracy"]

        # 先写一个表头，训练过程中填充具体值
        with open(self.csv_filename, mode="w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_columns)
            csv_writer.writeheader()

    def train(self, epoch: int):
        self.model.train()

        total_loss = 0
        correct_train = 0
        total_train = 0

        for index, (inputs, labels) in enumerate(self.train_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            self.logger.info(
                "Epoch [{}/{}]\tIteration [{}/{}]\tLoss: {:.6f}".format(
                    epoch,
                    self.num_epochs,  # TODO
                    index + 1,
                    len(self.train_dataloader),
                    loss.item(),
                )
            )

        # 计算平均损失和训练集准确率
        average_loss: float = total_loss / len(self.train_dataloader)
        train_accuracy: float = correct_train / total_train

        self.summary_writer.add_scalar(
            "train_loss", scalar_value=average_loss, global_step=epoch
        )
        self.summary_writer.add_scalar(
            "train_accu", scalar_value=train_accuracy, global_step=epoch
        )

        return {
            "Epoch": epoch,
            "Train Loss": average_loss,
            "Train Accuracy": train_accuracy,
        }

    def val(self, epoch):
        """验证函数"""
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total

        self.logger.info("Val Accuracy: {:.4f}\n".format(accuracy))
        self.summary_writer.add_scalar(
            "val_accu", scalar_value=accuracy, global_step=epoch
        )

        return {"Val Accuracy": accuracy}

    @staticmethod
    def create_summary_writer(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.summary_writer = SummaryWriter(comment=f"_{self.file_name}")
            try:
                return func(self, *args, **kwargs)
            finally:
                self.summary_writer.close()

        return wrapper

    @create_summary_writer
    def run(self):
        # 加载数据集
        data_handler = DataHandler(
            self.data_dir, self.batch_size, self.file_name, self.seed
        )
        self.train_dataloader, self.val_dataloader = data_handler.get_data_loaders()

        self.model.to(self.device)

        # 开始训练
        self.logger.info(f"Current LR: {self.lr:.8f}")
        for epoch in range(1, self.num_epochs + 1):
            train_accu = self.train(epoch)
            val_accu = self.val(epoch)

            train_accu.update(val_accu)

            # 保存到CSV文件
            with open(self.csv_filename, mode="a", newline="") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_columns)
                csv_writer.writerow(train_accu)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Current LR: {current_lr:.8f}")

        model_file_name = f"checkpoint/{self.model.name}/{self.file_name}.pth"

        # 保存权重文件
        torch.save(self.model.state_dict(), model_file_name)
        print(f"Training complete, model saved as '{model_file_name}'")

    @staticmethod
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

    @staticmethod
    def set_logger(filename):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("logger")

        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger
