# -*- encoding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn.metrics as sm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from IPython.display import clear_output

from . import models
from .dataset import GradCAMDataLoader, ShapDataLoader, TestDataLoader


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

    def load_dataloader(self, batch_size=None):
        raise NotImplementedError

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

        self.auc = 0
        self.ap = 0

        self.figure_save_path = f"./output/image/metrics/{self.file_name}"
        os.makedirs(self.figure_save_path, exist_ok=True)

    def load_test_dataloader(self, batch_size):
        test_dataset_loader = TestDataLoader(self.file_name)
        return DataLoader(test_dataset_loader, batch_size)

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

        plt.show()

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
        self.auc = sm.auc(fpr, tpr)
        print(f"AUC: {self.auc}")

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

        plt.show()

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
        self.ap = sm.average_precision_score(
            self.true_label, self.positive_probability, pos_label=1
        )
        print(f"AP: {self.ap}")

        fig, ax = plt.subplots(figsize=(5, 4))
        plt.plot(precision, recall, label="Logistic", linewidth=1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        # 设置y轴标签格式
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

        # 设置x轴和y轴刻度
        # ax.set_xticks([i * 0.1 for i in range(11)])
        ax.set_yticks([i * 0.1 for i in range(11)])

        plt.show()

        plt.savefig(
            os.path.join(self.figure_save_path, "pr_curve.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def confusion_matrix(self):
        conf_matrix = sm.confusion_matrix(self.true_label, self.pred_label)

        plt.figure(figsize=(5, 4))
        sns.set_theme(font_scale=1.3)

        ax = sns.heatmap(
            conf_matrix, annot=True, fmt="g", cmap="Blues", vmin=0, vmax=100
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # use matplotlib.colorbar.Colorbar object
        # cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        # cbar.ax.tick_params(labelsize=15)

        plt.show()
        sns.reset_defaults()

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
        self.cam_dataset_loader = self.load_dataloader()

        self.target_layer = self.get_target_layer()

        self.cam_save_path = f"./output/image/gradcam/{self.file_name}"
        os.makedirs(self.cam_save_path, exist_ok=True)

    def load_dataloader(self):
        return GradCAMDataLoader(self.file_name)

    def get_target_layer(self):
        if isinstance(self.model, models.ResNet18):
            return self.model.model.layer4[-1]
        if isinstance(self.model, models.ResNet50):
            return self.model.model.layer4[-1]
        elif isinstance(self.model, models.VGG16):
            return self.model.model.features[-1]
        elif isinstance(self.model, models.VGG19):
            return self.model.model.features[-1]
        elif isinstance(self.model, models.GoogLeNet):
            return self.model.model.inception5b
        elif isinstance(self.model, models.AlexNet):
            return self.model.model.features[-4]

    def img_overlay_mask(self, image):
        """生成单张图的 Grad-CAM 图像"""

        input_tensor = image / 1.0

        with SmoothGradCAMpp(self.model, self.target_layer) as cam_extractor:
            # Preprocess your data and feed it to the model
            out = self.model(input_tensor.unsqueeze(0))
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

        plt.savefig(
            os.path.join(self.cam_save_path, "cam_example.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def cam_examples(self):
        img_dict = {}
        labels = [1, 0]

        for label in labels:
            indices = [
                i for i, (_, l) in enumerate(self.cam_dataset_loader) if l == label
            ]
            label_name = "stained" if label == 1 else "non-stained"

            for idx, i in enumerate(indices[:6]):
                image, _ = self.cam_dataset_loader[i]
                title = f"({idx+1}) {label_name}"
                img_dict[title] = image.permute(1, 2, 0).numpy()
                img_dict[title + " (CAM)"] = self.img_overlay_mask(image)

        self.subplot(img_dict, ncols=6)

    def cam_test_dataset(self):
        tqdm_iterator = tqdm(range(len(self.cam_dataset_loader)), desc="Processing")

        for i in tqdm_iterator:
            image, label = self.cam_dataset_loader[i]

            title = f"{i}_{'stained' if label == 1 else 'non-stained'}"
            output_path = f"{self.cam_save_path}/{title}.png"

            if not os.path.exists(output_path):
                grad_cam_plot = np.hstack(
                    (
                        image.permute(1, 2, 0).numpy(),
                        np.array(self.img_overlay_mask(image)),
                    )
                )
                self.save_plot(grad_cam_plot, output_path)

            tqdm_iterator.set_postfix_str(f"{title}", refresh=False)

    def save_plot(self, image, output_path):
        plt.figure(dpi=300)
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            output_path,
            format="png",
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close()


class Shap(ModelEval):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

        self.model = self.load_model_weights()
        self.shap_dataset_loader = ShapDataLoader(self.file_name)

        self.shap_save_path = f"./output/image/shap/{self.file_name}"
        os.makedirs(self.shap_save_path, exist_ok=True)

        self.class_names = ["Non-Stained", "Stained"]

    @staticmethod
    def nhwc_to_nchw(x: torch.Tensor):
        if x.dim() == 4:
            x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
        return x

    @staticmethod
    def nchw_to_nhwc(x: torch.Tensor):
        if x.dim() == 4:
            x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
        return x

    def transform(self, image):
        img_transform = transforms.Compose(
            [
                transforms.Lambda(self.nhwc_to_nchw),
                transforms.Resize(size=(224, 224)),
                # transforms.ToTensor(),
                transforms.Lambda(lambda x: x * (1 / 255)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]
                ),
                transforms.Lambda(self.nchw_to_nhwc),
            ]
        )
        return img_transform(image)

    def inv_transform(self, image):
        image_transform = transforms.Compose(
            [
                transforms.Lambda(self.nhwc_to_nchw),
                transforms.Resize((224, 224)),
                transforms.Lambda(self.nchw_to_nhwc),
            ]
        )
        return image_transform(image)

    def predict(self, image):
        image = self.nhwc_to_nchw(torch.Tensor(image)).to(
            self.device
        )  # [1, 3, 224, 224]
        return self.model(image)

    def shap(
        self, image_original, image_trans, save=True, batch_size=50, n_evals=15000
    ):
        masker_blur = shap.maskers.Image(
            "blur(16, 16)", image_trans[0].shape
        )  # (1, 150528)

        explainer = shap.Explainer(
            self.predict, masker_blur, output_names=self.class_names
        )

        shap_values = explainer(
            image_trans, max_evals=n_evals, batch_size=batch_size, outputs=[0, 1]
        )

        shap_values.data = self.inv_transform(image_original).numpy()[0]
        shap_values.values = [
            val for val in np.moveaxis(shap_values.values[0], -1, 0)
        ]  # shap值热力图

        show = False if save is True else True

        shap.image_plot(
            shap_values=shap_values.values,
            pixel_values=shap_values.data,
            labels=shap_values.output_names,
            show=show,  # TODO: 辅助保存可视化结果
        )

    def one_sample_shap(self, save=False, image=None, label=None):
        """
        image: 传入Tensor，若未指定，则从测试集中抽取
        """
        if image is None or label is None:
            image, label = next(self.shap_dataset_loader)

        image_original = self.nchw_to_nhwc(image).unsqueeze(0)  # [1, 512, 512, 3]
        image_trans = self.transform(image_original)  # [1, 224, 224, 3]

        predict_label = torch.max(self.predict(image_trans).data, 1)[1].item()

        print(f"==>> True label: {label}")
        print(f"==>> Predict label: {predict_label}")

        plt.figure(dpi=300)
        self.shap(image_original, image_trans, save)

    def shap_test_dataset(self):
        for i, (image, label) in enumerate(self.shap_dataset_loader):
            image_original = self.nchw_to_nhwc(image).unsqueeze(0)  # [1, 512, 512, 3]
            image_trans = self.transform(image_original)  # [1, 224, 224, 3]

            predict_label = torch.max(self.predict(image_trans).data, 1)[1].item()
            output_path = f"{self.shap_save_path}/{i+1}_{label}_{predict_label}.png"

            if not os.path.exists(output_path):
                self.shap(image_original, image_trans)
                plt.savefig(output_path)
                plt.close()

            clear_output(wait=True)
