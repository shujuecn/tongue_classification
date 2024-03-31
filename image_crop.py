#!~/miniconda3/envs/torch-env/bin/python
# -*- encoding: utf-8 -*-

import os
import glob
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def resize_image(image_array: np.ndarray, target_size=(512, 512)):
    """
    重塑并填充图像尺寸，确保图像大小为目标尺寸
    :param image_array: 图像数组
    :param target_size: 目标尺寸，默认为 (512, 512)
    :return: 重塑并填充后的图像
    """
    image = Image.fromarray(image_array)
    width, height = image.size
    new_width = (
        target_size[0] if width > height else int(width * (target_size[0] / height))
    )
    new_height = (
        target_size[1] if height > width else int(height * (target_size[1] / width))
    )
    image = image.resize((new_width, new_height))
    image = ImageOps.pad(image, target_size, color=(0, 0, 0))

    return image


def crop_image(image_array: np.ndarray):
    """
    剪裁图像，去除黑边
    :param image_array: 图像数组
    :return: 剪裁后的图像
    """
    # 非全0的行向量
    row_sum: np.ndarray = np.sum(image_array, axis=(1, 2), keepdims=True)
    row_index: np.ndarray = np.any(row_sum != 0, axis=1).reshape(-1)

    # 非全0的列向量
    col_sum: np.ndarray = np.sum(image_array, axis=(0, 2), keepdims=True)
    col_index: np.ndarray = np.any(col_sum != 0, axis=0).reshape(-1)

    # 去掉黑边后的图像
    croped_image = image_array[row_index, :, :][:, col_index, :]

    return resize_image(croped_image)


def main(dataset, output):
    # 删除 .DS_Store
    _ = [os.remove(file) for file in glob.glob("**/.DS_Store", recursive=True)]

    for root, _, files in os.walk(dataset):
        dir_name = os.path.basename(root)
        tqdm_iterator = tqdm(files, desc=f"Processing {dir_name}", unit="file")

        for file in tqdm_iterator:
            # 新图片
            image_path = os.path.join(root, file)
            image_array = np.array(Image.open(image_path))
            croped_file = crop_image(image_array)

            # 新文件名
            croped_file_name = file.strip().lower()
            croped_file_path = os.path.join(output, dir_name, croped_file_name)

            # 导出
            os.makedirs(os.path.join(output, dir_name), exist_ok=True)
            croped_file.save(croped_file_path)

            # 更新进度条并且覆盖旧的print信息
            tqdm_iterator.set_postfix_str(f"{file} 处理成功", refresh=False)


if __name__ == "__main__":
    dataset = "./original_tongue_dataset"
    os.makedirs(output_path := "./croped_images", exist_ok=True)

    main(dataset, output_path)
