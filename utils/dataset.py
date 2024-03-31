import random
import glob
import csv

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class TrainDataLoader(Dataset):
    def __init__(self, train_dataset, train_labels) -> None:
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(
                    lambda img: img.rotate(random.choice([0, 90, 180, 270]))
                ),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]
                # ),
            ]
        )
        self.train_dataset = train_dataset
        self.train_labels = train_labels

    def __getitem__(self, index):
        image = Image.open(self.train_dataset[index])
        image = self.train_transform(image)
        label = self.train_labels[index]
        return image, label

    def __len__(self):
        return len(self.train_dataset)


class ValDataLoader(Dataset):
    def __init__(self, val_dataset, val_labels):
        self.val_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)), transforms.ToTensor()]
        )
        self.val_dataset = val_dataset
        self.val_labels = val_labels

    def __getitem__(self, index):
        image = Image.open(self.val_dataset[index])
        image = self.val_transform(image)
        label = self.val_labels[index]
        return image, label

    def __len__(self):
        return len(self.val_dataset)


class TestDataLoader(Dataset):
    def __init__(self, file_name: str):
        self.test_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)), transforms.ToTensor()]
        )
        self.test_file_name = f"./croped_images/split_info/{file_name}/test.csv"
        with open(self.test_file_name, "r") as file:
            self.data = list(csv.reader(file))

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path)
        image = self.test_transform(image)

        return image, int(label)

    def __len__(self):
        return len(self.data)


class DataHandler:
    def __init__(self, data_dir, batch_size, file_name, seed):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_name = file_name

        self.train_size = 1406
        self.val_size = 402
        self.test_size = 200

        self.seed = seed
        self.random_split(self.seed)
        self.save_split_info()

    def random_split(self, seed):
        """随机拆分数据"""
        data_list = glob.glob("**/*.jpg", recursive=True)
        random.seed(seed)
        random.shuffle(data_list)

        self.train_dataset = data_list[: self.train_size]
        self.val_dataset = data_list[self.train_size : -self.test_size]
        self.test_dataset = data_list[-self.test_size :]

        # 创建标签
        # 0: non-stained（非染苔）
        # 1: stained（染苔）
        self.train_labels = [0 if "non" in data else 1 for data in self.train_dataset]
        self.val_labels = [0 if "non" in data else 1 for data in self.val_dataset]
        self.test_labels = [0 if "non" in data else 1 for data in self.test_dataset]

    def save_split_info(self):
        """保存数据拆分信息到CSV文件"""
        self.dataset_dict = {
            "train": (self.train_dataset, self.train_labels),
            "val": (self.val_dataset, self.val_labels),
            "test": (self.test_dataset, self.test_labels),
        }

        for filename, (dataset, labels) in self.dataset_dict.items():
            with open(
                f"{self.data_dir}/split_info/{self.file_name}/{filename}.csv",
                mode="w",
            ) as csv_file:
                for image_path, label in zip(dataset, labels):
                    csv_file.write(f"{image_path}, {label}\n")

    def get_data_loaders(self):
        """创建 DataLoader"""
        train_dataset_loader = TrainDataLoader(self.train_dataset, self.train_labels)
        val_dataset_loader = ValDataLoader(self.val_dataset, self.val_labels)

        return (
            DataLoader(train_dataset_loader, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_dataset_loader, batch_size=self.batch_size),
        )
