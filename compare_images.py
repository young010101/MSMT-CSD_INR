import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_and_analyze_image(file_path):
    """加载nifti图像并计算统计特征"""
    img = nib.load(file_path)
    data = img.get_fdata()
    
    stats = {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "non_zero": np.mean(data != 0),
        "shape": data.shape
    }
    
    return data, stats

def compare_slices(images, titles, slice_idx=None, coeff_idx=0, save_prefix=""):
    """比较不同图像的同一个切片的特定系数"""
    if slice_idx is None:
        slice_idx = images[0].shape[2] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    vmin = min(np.min(img[:, :, slice_idx, coeff_idx]) for img in images)
    vmax = max(np.max(img[:, :, slice_idx, coeff_idx]) for img in images)
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        slice_data = img[:, :, slice_idx, coeff_idx]
        im = axes[idx].imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"{title}\nSlice {slice_idx}, Index {coeff_idx}")
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_slice_comparison_{coeff_idx}.png')
    plt.close()

def main():
    base_dir = Path("runs/training_strategies")
    ratios = ["30", "40", "50", "60"]
    
    # 存储所有图像的统计信息
    coeff_images = []
    grad_images = []
    
    print("图像统计信息比较：")
    print("=" * 50)
    
    # 分析系数图像
    print("\n系数图像 (Coefficients) 统计信息：")
    print("-" * 50)
    for ratio in ratios:
        coeff_file = list((base_dir / f"random_{ratio}").glob("coeffs_2025*.nii.gz"))[0]
        data, stats = load_and_analyze_image(coeff_file)
        coeff_images.append(data)
        
        print(f"\n{ratio}% 训练集结果:")
        print(f"形状: {stats['shape']}")
        print(f"均值: {stats['mean']:.6f}")
        print(f"标准差: {stats['std']:.6f}")
        print(f"最小值: {stats['min']:.6f}")
        print(f"最大值: {stats['max']:.6f}")
        print(f"非零比例: {stats['non_zero']:.6f}")
    
    # 分析梯度图像
    print("\n梯度图像 (Gradients) 统计信息：")
    print("-" * 50)
    for ratio in ratios:
        grad_file = list((base_dir / f"random_{ratio}").glob("grads_2025*.nii.gz"))[0]
        data, stats = load_and_analyze_image(grad_file)
        grad_images.append(data)
        
        print(f"\n{ratio}% 训练集结果:")
        print(f"形状: {stats['shape']}")
        print(f"均值: {stats['mean']:.6f}")
        print(f"标准差: {stats['std']:.6f}")
        print(f"最小值: {stats['min']:.6f}")
        print(f"最大值: {stats['max']:.6f}")
        print(f"非零比例: {stats['non_zero']:.6f}")
    
    # 比较系数图像
    print("\n生成系数图像对比...")
    for coeff_idx in [0, 22, 44]:  # 比较第一个、中间和最后一个系数
        compare_slices(coeff_images, [f"{r}% Training" for r in ratios], 
                      coeff_idx=coeff_idx, save_prefix="coeffs")
    
    # 比较梯度图像
    print("生成梯度图像对比...")
    for grad_idx in [0, 44, 89]:  # 比较不同的梯度方向
        compare_slices(grad_images, [f"{r}% Training" for r in ratios], 
                      coeff_idx=grad_idx, save_prefix="grads")
    
    print("\n图像对比已保存：")
    print("- 系数图像对比: coeffs_slice_comparison_*.png")
    print("- 梯度图像对比: grads_slice_comparison_*.png")

if __name__ == "__main__":
    main() 