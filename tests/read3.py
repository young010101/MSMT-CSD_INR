import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 读取NIFTI文件
img = nib.load('../runs/mwu100408_von/grads_test.nii.gz')
data = img.get_fdata()

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")

# 显示中间切片
middle_slice = data.shape[2] // 2
plt.figure(figsize=(12, 4))

# 显示第一个时间点的三个视图
time_point = 0
plt.subplot(1, 3, 1)
plt.imshow(data[:, :, middle_slice, time_point], cmap='gray')
plt.title(f'Axial slice (z={middle_slice})')

plt.subplot(1, 3, 2)
plt.imshow(data[:, data.shape[1] // 2, :, time_point], cmap='gray')
plt.title('Sagittal slice')

plt.subplot(1, 3, 3)
plt.imshow(data[data.shape[0] // 2, :, :, time_point], cmap='gray')
plt.title('Coronal slice')

plt.tight_layout()
plt.show()
