# -*- encoding: utf-8 -*-

import contextlib
import os
import glob
from shutil import rmtree

if __name__ == "__main__":
    files_to_delete = [
        "./checkpoint",
        "./logs",
        "./output",
        "./runs",
        "./croped_images/split_info",
        "./.vscode",
        "./utils/__pycache__",
        "./.ipynb_checkpoints",
    ]

    ds_store_files = glob.glob("**/.DS_Store", recursive=True)

    print("The following files and directories will be deleted:")
    for file in files_to_delete:
        print(f"- {file}")
    print("\nThe following .DS_Store files will be deleted:")
    for file in ds_store_files:
        print(f"- {file}")

    # 获取用户确认
    confirm = input("\nAre you sure you want to delete these files? (yes/no): ").lower()

    if confirm == "yes":
        # 删除目录
        for file in files_to_delete:
            with contextlib.suppress(FileNotFoundError):
                rmtree(file)
        # 删除 .DS_Store 文件
        for file in ds_store_files:
            with contextlib.suppress(FileNotFoundError):
                os.remove(file)
        print("Files deleted successfully.")
    else:
        print("Deletion aborted.")
