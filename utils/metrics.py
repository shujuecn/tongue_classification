# -*- encoding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sm
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from . import models
from .dataset import TestDataLoader


class ModelEval:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        timestamp, model_name, _, num_epoch, _, batch_size, _, lr = file_name.split("_")

        self.timestamp = timestamp
        self.model_name = model_name
        self.num_epoch = int(num_epoch)
        self.batch_size = int(batch_size)
        self.lr = float(lr)

        self.device = self.get_device()

        self.checkpoint_path = f"./checkpoint/{self.model_name}/{self.file_name}.pth"
        self.csv_filename = f"./output/train_info/{self.file_name}.csv"
        self.figure_save_path = f"./output/image/metrics/{self.file_name}"

        os.makedirs(self.figure_save_path, exist_ok=True)

    def load_model_weights(self, device=None):
        """加载模型权重，评估模式"""
        model = models.get_model(self.model_name)

        model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )

        if device != "cpu":
            model.to(self.device)

        model.eval()
        return model

    def load_test_dataloader(self, batch_size):
        test_dataset_loader = TestDataLoader(self.file_name)
        return DataLoader(test_dataset_loader, batch_size)

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


class ClassificationEvaluator(ModelEval):

    def __init__(self, file_name) -> None:
        super().__init__(file_name)

        self.pred_prob: torch.Tensor
        self.pred_label: torch.Tensor
        self.true_label: torch.Tensor
        self.positive_probability: torch.Tensor

        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def predict(self):
        """预测图像类别并返回概率"""

        model = self.load_model_weights()
        test_dataloader = self.load_test_dataloader(self.batch_size)

        pred_prob = []
        pred_label = []
        true_label = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1).squeeze()
                predicted_labels = probabilities.argmax(1)

                pred_prob.append(probabilities)
                pred_label.append(predicted_labels)
                true_label.append(labels)

        self.pred_prob = torch.cat(pred_prob).cpu()
        self.pred_label = torch.cat(pred_label).cpu()
        self.true_label = torch.cat(true_label).cpu()

        self.positive_probability = self.pred_prob[:, 1]

    def evaluation_score(self):
        self.accuracy = sm.accuracy_score(self.true_label, self.pred_label)
        self.precision = sm.precision_score(self.true_label, self.pred_label)
        self.recall = sm.recall_score(self.true_label, self.pred_label)
        self.f1 = sm.f1_score(self.true_label, self.pred_label)

        return {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1-Score": self.f1,
        }

    def training_curve(self):
        """绘制曲线"""

        # 读取CSV文件
        df = pd.read_csv(self.csv_filename)

        # 提取数据
        epochs = df["Epoch"]
        train_loss = df["Train Loss"]
        train_accuracy = df["Train Accuracy"]
        val_accuracy = df["Val Accuracy"]

        # 绘制折线图
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color=color)
        ax1.plot(
            epochs,
            train_loss,
            color=color,
            linestyle="--",
            label="Train Loss",
            linewidth=1,
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
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        # x轴刻度
        ax1.set_xticks(range(0, len(epochs) + 1, 5))
        ax1.set_xticklabels(range(0, len(epochs) + 1, 5))

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
            f"Batch size: {int(self.batch_size)}\n"
            f"Learning rate: {float(self.lr)}\n"
            f"Average training accuracy: {train_accuracy.mean():.4f}\n"
            f"Average verification accuracy: {val_accuracy.mean():.4f}\n"
            f"Test Accuracy: {self.accuracy:.4f}"
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
            os.path.join(self.figure_save_path, "training_curve.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def roc_curve(self):
        fpr, tpr, thresholds = sm.roc_curve(
            self.true_label, self.positive_probability, pos_label=1
        )
        print(f"AUC: {sm.auc(fpr, tpr)}")

        fig, ax = plt.subplots(figsize=(5, 4))
        plt.plot(fpr, tpr, linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.title("ROC curve")

        plt.plot([0, 1], [0, 1], "r--", linewidth=1)  # 对角线，红色虚线

        # 设置y轴标签格式
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

        # 设置x轴和y轴刻度
        ax.set_xticks([i * 0.1 for i in range(11)])
        ax.set_yticks([i * 0.1 for i in range(11)])

        plt.savefig(
            os.path.join(self.figure_save_path, "roc_curve.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def pr_curve(self):
        precision, recall, threshold = sm.precision_recall_curve(
            self.true_label, self.positive_probability, pos_label=1
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        plt.plot(precision, recall, label="Logistic", linewidth=1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        # 设置y轴标签格式
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

        # 设置x轴和y轴刻度
        # ax.set_xticks([i * 0.1 for i in range(11)])
        ax.set_yticks([i * 0.1 for i in range(11)])

        plt.savefig(
            os.path.join(self.figure_save_path, "pr_curve.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def confusion_matrix(self):

        conf_matrix = sm.confusion_matrix(self.true_label, self.pred_label)

        plt.figure(figsize=(5, 4))

        # vmin, vmax = 0, 100
        ax = sns.heatmap(
            conf_matrix, annot=True, fmt="g", cmap="Blues", vmin=0, vmax=100
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        plt.savefig(
            os.path.join(self.figure_save_path, "confusion_matrix.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )


class GradCAM(ModelEval):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

        self.model = self.load_model_weights("cpu")
        self.test_dataset_loader = TestDataLoader(file_name)

        self.target_layer = self.get_target_layer()

        self.cam_save_path = f"./output/image/gradcam/{self.file_name}"
        os.makedirs(self.cam_save_path, exist_ok=True)

    def get_target_layer(self):
        if isinstance(self.model, models.ResNet50):
            return self.model.model.layer4[-1].conv3
        elif isinstance(self.model, models.VGG16):
            return self.model.features[-1]
        elif isinstance(self.model, models.GoogLeNet):
            return self.model.inception5b
        elif isinstance(self.model, models.AlexNet):
            return self.model.features[-4]

    def img_overlay_mask(self, image):
        """生成单张图的 Grad-CAM 图像"""

        with SmoothGradCAMpp(self.model, self.target_layer) as cam_extractor:
            # Preprocess your data and feed it to the model
            out = self.model(image.unsqueeze(0))
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        return overlay_mask(
            to_pil_image(image),
            to_pil_image(activation_map[0].squeeze(0), mode="F"),
            alpha=0.5,
        )

    def subplot(self, image_dict, nrows=None, ncols=None, title=None):
        """多子图排版"""
        # nrows = 1 if not nrows else nrows
        # ncols = len(image_dict) if not ncols else ncols

        if nrows is None and ncols is None:
            nrows, ncols = 1, len(image_dict)
        elif nrows is None:
            nrows = (len(image_dict) + ncols - 1) // ncols
        elif ncols is None:
            ncols = (len(image_dict) + nrows - 1) // nrows

        plt.figure(figsize=(10, 7), dpi=300)

        for i, (k, v) in enumerate(image_dict.items()):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(v)
            plt.title(k, size=8)
            plt.axis("off")

        plt.tight_layout()

        if title:
            plt.suptitle(title, y=1.03)

    def cam_examples(self):

        img_dict = {}

        for i in range(12):
            image, label = self.test_dataset_loader[i]

            title = f"({i+1}) {'stained' if label == 1 else 'non-stained'}"
            img_dict[title] = image.permute(1, 2, 0).numpy()
            img_dict[title + " (CAM)"] = self.img_overlay_mask(image)

        self.subplot(img_dict, ncols=6)

    def cam_test_dataset(self):
        tqdm_iterator = tqdm(range(len(self.test_dataset_loader)), desc="Processing")

        for i in tqdm_iterator:
            image, label = self.test_dataset_loader[i]

            grad_cam_plot = np.hstack(
                (
                    image.permute(1, 2, 0).numpy(),
                    np.array(self.img_overlay_mask(image)) / 255.0,
                )
            )

            title = f"{i}_{'stained' if label == 1 else 'non-stained'}"
            self.save_plot(grad_cam_plot, title)
            tqdm_iterator.set_postfix_str(f"{title}", refresh=False)

    def save_plot(self, image, title):
        plt.figure(dpi=300)
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            f"{self.cam_save_path}/{title}.png",
            format="png",
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close()