import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

output_dir = Path('runs/mwu100408_von')
ii_files = list(output_dir.glob('*.nii.gz'))

print('Found:', [f.name for f in ii_files])

for nii_path in ii_files:
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    stats = f'{nii_path.name}: shape={data.shape}, min={np.min(data)}, max={np.max(data)}, mean={np.mean(data)}'
    print(stats)
    if np.isnan(data).any() or np.isinf(data).any():
        print(f'  ⚠️  {nii_path.name} 包含NaN或Inf！')
    # 显示中间切片
    if data.ndim == 3:
        mid = tuple(s//2 for s in data.shape)
        slice_img = data[mid[0], :, :]
    elif data.ndim == 4:
        mid = tuple(s//2 for s in data.shape[:3])
        slice_img = data[mid[0], :, :, 0]
    else:
        slice_img = data
    plt.imshow(slice_img, cmap='gray')
    plt.title(nii_path.name)
    plt.axis('off')
    out_png = output_dir / f'{nii_path.stem}_mid.png'
    plt.savefig(str(out_png), bbox_inches='tight')
    plt.close()
    print(f'Saved {out_png}') 