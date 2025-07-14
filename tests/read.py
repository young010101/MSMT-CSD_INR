import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 读取NIFTI文件
img = nib.load('../runs/mwu100408_von/coeffs_test.nii.gz')
data = img.get_fdata()

print(f"数据维度: {data.shape}")
print(f"数据类型: {data.dtype}")

# 显示中间切片（第3个维度的中间层）
middle_slice = data.shape[2] // 2
plt.figure(figsize=(12, 4))

# 显示第一个时间点的三个视图
time_point = 0
plt.subplot(1, 3, 1)
plt.imshow(data[:, :, middle_slice, time_point], cmap='gray')
plt.title(f'轴向切片 (z={middle_slice})')

plt.subplot(1, 3, 2)
plt.imshow(data[:, data.shape[1] // 2, :, time_point], cmap='gray')
plt.title('矢状切片')

plt.subplot(1, 3, 3)
plt.imshow(data[data.shape[0] // 2, :, :, time_point], cmap='gray')
plt.title('冠状切片')

plt.tight_layout()
plt.show()
