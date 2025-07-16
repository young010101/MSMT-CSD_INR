import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

nii_path = "runs/mwu100408_250716/grads_test.nii.gz"
img = nib.load(nii_path)
data = img.get_fdata()

# 只取第一个通道（如有需要可改为其他通道）
data_3d = data[..., 0]

# 创建输出文件夹
os.makedirs("slice_images", exist_ok=True)

# 取不同方向的中间切面
slices = {
    "axial": data_3d[:, :, data_3d.shape[2] // 2],      # 横断面 (x, y)
    "sagittal": data_3d[data_3d.shape[0] // 2, :, :],      # 矢状面 (y, z)
    "coronal": data_3d[:, data_3d.shape[1] // 2, :]        # 冠状面 (x, z)
}

for name, slice2d in slices.items():
    plt.figure(figsize=(6, 6))
    # 保证传入imshow的是二维数组
    if name == "sagittal":
        plt.imshow(np.rot90(slice2d), cmap="gray")
    elif name == "coronal":
        plt.imshow(np.rot90(slice2d), cmap="gray")
    else:
        plt.imshow(slice2d, cmap="gray", origin="lower")
    plt.title(name)
    plt.axis("off")
    plt.savefig(f"slice_images/{name}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

print("切面图片已保存到 slice_images 文件夹。")