import os
import torch
from torchvision.io.image import read_image
from torchvision.transforms.functional import resize, center_crop

batch_size = 1403
h, w = 224, 224
batch = torch.zeros(batch_size, 3, h, w, dtype=torch.uint8)

data_dir = "./datasets/train/"
i = 0

for root, _, files in os.walk(data_dir):
    for file in files:
        if not file.endswith(("jpg", "png")):
            continue
        img_arr = read_image(os.path.join(root, file))
        img_arr = center_crop(resize(img_arr, (256, 256), antialias=True), 224)

        batch[i] = img_arr
        i += 1

batch = batch.float()
batch /= 255.0

n_channels = batch.shape[1]
means, stds = [], []
for c in range(n_channels):
    mean = torch.mean(batch[:, c]).item()
    means.append(round(mean, 3))
    std = torch.std(batch[:, c]).item()
    stds.append(round(std, 3))
    # batch[:, c] = (batch[:, c] - mean) / std

print(f"means={means}")
print(f"stds={stds}")
