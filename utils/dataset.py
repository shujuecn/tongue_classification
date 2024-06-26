import random
import glob
import csv

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.io.image import read_image


class EmptyDataLoader(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def get_transform(self):
        raise NotImplementedError

    def __getitem__(self, index):
        image = Image.open(self.dataset[index])
        transform = self.get_transform()
        image = transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.dataset)


class TrainDataLoader(EmptyDataLoader):
    def __init__(self, dataset, labels) -> None:
        super().__init__(dataset, labels)

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(
                    lambda img: img.rotate(random.choice([0, 90, 180, 270]))
                ),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]
                ),
            ]
        )


class ValDataLoader(EmptyDataLoader):
    def __init__(self, dataset, labels) -> None:
        super().__init__(dataset, labels)

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]
                ),
            ]
        )


class TestDataLoader(EmptyDataLoader):
    def __init__(self, file_name: str):
        self.test_file_name = f"./croped_images/split_info/{file_name}/test.csv"
        with open(self.test_file_name, "r") as file:
            self.data = list(csv.reader(file))

        self.dataset = [d[0] for d in self.data]
        self.labels = [int(d[1]) for d in self.data]

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]
                ),
            ]
        )


class GradCAMDataLoader(Dataset):
    def __init__(self, file_name: str):
        self.file_path = f"./croped_images/split_info/{file_name}/test.csv"
        with open(self.file_path, "r") as file:
            self.data = list(csv.reader(file))

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = read_image(image_path)

        return image, int(label)

    def __len__(self):
        return len(self.data)


class ShapDataLoader(GradCAMDataLoader):
    def __init__(self, file_name: str):
        super().__init__(file_name)
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self):
            image, label = self[self.index]
            self.index += 1
            return image, label
        else:
            raise StopIteration()


class DataHandler:
    def __init__(
        self,
        data_dir,
        batch_size,
        file_name,
        seed,
        data_path,
        train_size,
        val_size,
        test_size,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_name = file_name

        self.data_path = data_path
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        self.seed = seed
        self.random_split(self.seed)
        self.save_split_info()

    def random_split(self, seed):
        """随机拆分数据"""
        # 修复读取非训练图片错误
        data_list = glob.glob(f"{self.data_path}/**/*.jpg", recursive=True)
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
