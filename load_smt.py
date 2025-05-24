import nibabel as nib
import numpy as np

# 路径配置
nii_path = "Delta_19/qb_diff_real_delta_19.nii"
bvals_path = "Delta_19/bvals_19.txt"

# 读取 NIfTI 图像
img = nib.load(nii_path)
data = img.get_fdata()  # shape: (108, 108, 70, 430)

# 读取 bvals
bvals = np.loadtxt(bvals_path)

# 找到所有唯一的 b 值
unique_bvals = np.unique(bvals)

# 创建一个空列表来存放平均后的结果
averaged_volumes = []

# 对每个 b 值分组，计算对应图像的平均
for b in unique_bvals:
    indices = np.where(bvals == b)[0]
    mean_volume = np.mean(data[..., indices], axis=3)
    averaged_volumes.append(mean_volume)

# 将平均结果堆叠成新的 4D 图像
averaged_data = np.stack(averaged_volumes, axis=-1)

# 创建新图像对象（沿用原图像的仿射和头信息）
new_img = nib.Nifti1Image(averaged_data, affine=img.affine, header=img.header)

# 保存图像（可选）
nib.save(new_img, "qb_diff_real_delta_19_avg_by_b.nii")
