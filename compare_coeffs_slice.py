import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# 文件路径（请根据实际情况修改）
nii_files = {
    "全量训练": "runs/mwu100408_250716/coeffs_test.nii.gz",
    "70%欠采样": "runs/mwu100408_250716/coeffs_colfg8vl_random_sample.nii.gz",
    "奇偶分离": "runs/mwu100408_250716/coeffs_tcqivmvu_oddeven.nii.gz"
}

# 选择的slice编号
slice_id = 45

# 读取并存储每个策略的数据
data_dict = {}
for name, path in nii_files.items():
    img = nib.load(path)
    data = img.get_fdata()
    data_dict[name] = data
    print(f"{name}: 数据形状 {data.shape}")

# 1单slice对比图
print("\n==== 1 单slice对比图 ====")
slices = {}
for name, data in data_dict.items():
    if data.ndim == 4:
        data_3d = data[..., 0]  # 取第一个通道
    else:
        data_3d = data
    if data_3d.shape[2] > slice_id:
        slices[name] = data_3d[:, :, slice_id]
    else:
        print(f"{name} 文件z轴长度不足{slice_id+1}层，仅显示最后一层。")
        slices[name] = data_3d[:, :, -1]

# 拼图
plt.figure(figsize=(15, 5))
for i, (name, slice2d) in enumerate(slices.items()):
    plt.subplot(1, 3, i+1)
    plt.imshow(slice2d, cmap="gray", origin="lower")
    plt.title(f"{name}\nSlice {slice_id}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("compare_coeffs_slice.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. 数值统计对比
print("\n====2值统计对比 ====")
stats_data = {}
for name, data in data_dict.items():
    if data.ndim == 4:
        data_flat = data.flatten()
    else:
        data_flat = data.flatten()
    
    stats_data[name] = {
        "mean": np.mean(data_flat),
        "std": np.std(data_flat),
        "min": np.min(data_flat),
        "max": np.max(data_flat),
        "median": np.median(data_flat),
        "q25": np.percentile(data_flat, 25),
        "q75": np.percentile(data_flat, 75),
        "skewness": stats.skew(data_flat),
        "kurtosis": stats.kurtosis(data_flat)
    }
    print(f"\n{name} 统计信息:")
    for stat, value in stats_data[name].items():
        print(f"  {stat}: {value:.6f}")

#3. 统计对比表格
fig, ax = plt.subplots(figsize=(12, 8))
stats_df = []
for name, stats_dict in stats_data.items():
    stats_dict['method'] = name
    stats_df.append(stats_dict)

# 创建热力图
stats_matrix = np.array([[stats_data[name][stat] for stat in ['mean', 'std', 'min', 'max', 'median']] 
                         for name in stats_data.keys()])
plt.figure(figsize=(10, 6))
sns.heatmap(stats_matrix, 
            xticklabels=['mean', 'std', 'min', 'max', 'median'],
            yticklabels=list(stats_data.keys()),
            annot=True, fmt='.4f', cmap='viridis')
plt.title('数值统计对比热力图')
plt.tight_layout()
plt.savefig("stats_comparison_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

#4slice对比（取3个不同slice）
print("\n====3. 多slice对比 ====")
slice_ids = [slice_id//3, slice_id, 2*slice_id//3]  # 前中后三个slice
fig, axes = plt.subplots(len(slice_ids), len(data_dict), figsize=(15,12))
if len(slice_ids) == 1:
    axes = axes.reshape(1, len(slice_ids))

for i, sid in enumerate(slice_ids):
    for j, (name, data) in enumerate(data_dict.items()):
        if data.ndim == 4:
            data_3d = data[..., 0]
        else:
            data_3d = data
        if data_3d.shape[2] > sid:
            slice_data = data_3d[:, :, sid]
        else:
            slice_data = data_3d[:, :, -1]
        
        axes[i, j].imshow(slice_data, cmap="gray", origin="lower")
        axes[i, j].set_title(f"{name}\nSlice {sid}")
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig("multi_slice_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# 5通道对比（如果有多通道）
print("\n==== 4对比 ====")
for name, data in data_dict.items():
    if data.ndim == 4 and data.shape[-1] > 1:
        print(f"\n{name} 有 {data.shape[-1]} 个通道")
        fig, axes = plt.subplots(1, min(3, data.shape[-1]), figsize=(15, 5))
        if data.shape[-1] == 1:
            axes = [axes]
        
        for ch in range(min(3, data.shape[-1])):
            if data.shape[2] > slice_id:
                slice_data = data[:, :, slice_id, ch]
            else:
                slice_data = data[:, :, -1, ch]
            
            axes[ch].imshow(slice_data, cmap="gray", origin="lower")
            axes[ch].set_title(f"Channel {ch}")
            axes[ch].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"channel_comparison_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

#6差异分析
print("\n==== 5. 差异分析 ====")
# 以全量训练为基准，计算其他方法的差异
baseline_name = "全量训练"
baseline_data = data_dict[baseline_name]
if baseline_data.ndim == 4:
    baseline_3d = baseline_data[..., 0]
else:
    baseline_3d = baseline_data

diff_fig, diff_axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (name, data) in enumerate(data_dict.items()):
    if name == baseline_name:
        continue
    
    if data.ndim == 4:
        data_3d = data[..., 0]
    else:
        data_3d = data
    
    # 确保形状一致
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(baseline_3d.shape, data_3d.shape))
    baseline_slice = baseline_3d[:min_shape[0], :min_shape[1], :min_shape[2]]
    data_slice = data_3d[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # 计算差异
    diff = data_slice - baseline_slice
    abs_diff = np.abs(diff)
    
    # 显示差异图
    if i < 2:  # 只显示前两个差异
        diff_axes[i].imshow(diff[:, :, slice_id], cmap='RdBu_r', origin="lower")
        diff_axes[i].set_title(f"{name} - {baseline_name}\nSlice {slice_id}")
        diff_axes[i].axis("off")
        
        print(f"\n{name} vs {baseline_name} 差异统计:")
        print(f"  平均差异: {np.mean(diff):.6f}")
        print(f"  平均绝对差异: {np.mean(abs_diff):.6f}")
        print(f"  最大绝对差异: {np.max(abs_diff):.6f}")
        print(f"  标准差: {np.std(diff):.6f}")

plt.tight_layout()
plt.savefig("difference_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# 7. 质量评估指标
print("\n====6 质量评估指标 ====")
for name, data in data_dict.items():
    if name == baseline_name:
        continue
    
    if data.ndim == 4:
        data_3d = data[..., 0]
    else:
        data_3d = data
    
    # 确保形状一致
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(baseline_3d.shape, data_3d.shape))
    baseline_slice = baseline_3d[:min_shape[0], :min_shape[1], :min_shape[2]]
    data_slice = data_3d[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # 计算质量指标
    mse = np.mean((data_slice - baseline_slice) ** 2)
    mae = np.mean(np.abs(data_slice - baseline_slice))
    psnr = 20 * np.log10(np.max(baseline_slice) / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # 结构相似性 (SSIM的简化版本)
    mu_x = np.mean(baseline_slice)
    mu_y = np.mean(data_slice)
    sigma_x = np.std(baseline_slice)
    sigma_y = np.std(data_slice)
    sigma_xy = np.mean((baseline_slice - mu_x) * (data_slice - mu_y))
    
    ssim = ((2 * mu_x * mu_y + 1e-8) / ((mu_x**2 + mu_y**2 + 1e-8) * (sigma_x**2 + sigma_y**2 + 1e-8)))
    
    print(f"\n{name} vs {baseline_name} 质量评估:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.6f}")

#8 保存详细统计报告
print("\n====7生成详细报告 ====")
with open("comparison_report.txt", "w", encoding="utf-8") as f:
    f.write("训练策略对比分析报告\n")
    f.write("=" * 50 + "\n")
    
    f.write("1据基本信息:\n")
    for name, data in data_dict.items():
        f.write(f"  {name}: 形状 {data.shape}, 数据类型 {data.dtype}\n")
    
    f.write("\n2值统计对比:\n")
    for name, stats_dict in stats_data.items():
        f.write(f"\n  {name}:\n")
        for stat, value in stats_dict.items():
            f.write(f"    {stat}: {value:.6f}\n")
    
    f.write("\n3. 质量评估:\n")
    for name, data in data_dict.items():
        if name == baseline_name:
            continue
        # 这里可以添加质量评估的计算和写入

print("对比分析完成！生成的文件:")
print("- compare_coeffs_slice.png: 单slice对比图")
print("- stats_comparison_heatmap.png: 统计热力图")
print("- multi_slice_comparison.png: 多slice对比图")
print("- difference_analysis.png: 差异分析图")
print("- comparison_report.txt: 详细统计报告")
