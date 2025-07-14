import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt

n19 = nib.load('/data/users/cyang/notes/Research/genAI/diffusion/' + 'qb_diff_real_delta_19_avg_by_b.nii')
n49 = nib.load('/home/cyang/repos/MSMT-CSD_INR/' + 'qb_diff_real_delta_49_avg_by_b.nii')
mask = nib.load('/data/users/zzhou/GNC/Data/HC_030/Delta_19/brainmask.nii.gz')

print(mask.get_fdata().shape)
# 显示 mask的值
print(mask.get_fdata())

mask_4d = mask.get_fdata()[..., np.newaxis]

full_img = np.concatenate([n19.get_fdata()[..., 1:], n49.get_fdata()[..., 1:]], axis=-1)

full_img_masked = full_img * mask_4d
plt.imshow(full_img_masked[:, :, 35, 0])
plt.show()
print(f"小于0的像素：{np.sum(full_img_masked < 0)}")

print(full_img.shape)

plt.plot(full_img[50, 50, 35, :])
plt.show()

print(f"小于0的像素：{np.sum(full_img < 0)}")
