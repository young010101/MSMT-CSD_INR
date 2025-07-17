import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def load_gradient_data(file_path):
    """加载梯度数据文件"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        # 如果数据是多维的，计算每个体素的梯度幅值
        if len(data.shape) > 3:
            # 假设最后一维是梯度分量
            data = np.sqrt(np.sum(data**2, axis=-1))
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_odd_even_layers(data, title=""):
    """分析奇偶层的梯度分布"""
    n_layers = data.shape[2]  # 假设第三维是层
    odd_layers = data[:, :, 1::2]
    even_layers = data[:, :, 0::2]
    
    # 计算统计信息
    odd_mean = np.mean(odd_layers)
    even_mean = np.mean(even_layers)
    odd_std = np.std(odd_layers)
    even_std = np.std(even_layers)
    
    # 创建统计图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.boxplot([odd_layers.flatten(), even_layers.flatten()], labels=['Odd layers', 'Even layers'])
    plt.title(f'{title}\nOdd vs Even Layer Distribution')
    plt.ylabel('Gradient Magnitude')
    
    plt.subplot(122)
    layer_means = [np.mean(data[:,:,i]) for i in range(n_layers)]
    plt.plot(range(n_layers), layer_means, 'b-', label='Layer mean')
    plt.fill_between(range(n_layers), 
                    [m - np.std(data[:,:,i]) for i, m in enumerate(layer_means)],
                    [m + np.std(data[:,:,i]) for i, m in enumerate(layer_means)],
                    alpha=0.3)
    plt.title('Layer-wise Mean Gradient Magnitude')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Gradient Magnitude')
    plt.legend()
    
    plt.tight_layout()
    return {
        'odd_mean': odd_mean,
        'even_mean': even_mean,
        'odd_std': odd_std,
        'even_std': even_std
    }

def compare_gradient_maps(data_dict):
    """比较不同策略的梯度图"""
    n_strategies = len(data_dict)
    fig, axes = plt.subplots(n_strategies, 3, figsize=(15, 5*n_strategies))
    if n_strategies == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (strategy, data) in enumerate(data_dict.items()):
        if data is None:
            continue
            
        # 选择中间层进行显示
        mid_layer = data.shape[2] // 2
        
        # 显示中间层的梯度图
        im = axes[idx, 0].imshow(data[:, :, mid_layer], cmap='viridis')
        axes[idx, 0].set_title(f'{strategy}\nMiddle Layer Gradient Map')
        plt.colorbar(im, ax=axes[idx, 0])
        
        # 显示沿x方向的梯度分布
        mean_x = np.mean(data, axis=1)
        sns.heatmap(mean_x, ax=axes[idx, 1], cmap='viridis')
        axes[idx, 1].set_title('X-direction Average')
        
        # 显示沿y方向的梯度分布
        mean_y = np.mean(data, axis=0)
        sns.heatmap(mean_y, ax=axes[idx, 2], cmap='viridis')
        axes[idx, 2].set_title('Y-direction Average')
    
    plt.tight_layout()

def main():
    # 设置数据路径
    base_path = Path('runs/mwu100408_250716')
    files = {
        'Full Training': base_path / 'grads_test.nii.gz',
        '70% Sampling': base_path / 'grads_colfg8vl.nii.gz',
        'Odd-Even Split': base_path / 'grads_tcqivmvu.nii.gz'
    }
    
    # 加载数据
    data_dict = {name: load_gradient_data(str(path)) for name, path in files.items()}
    
    # 创建输出目录
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # 比较梯度图
    plt.figure(figsize=(15, 15))
    compare_gradient_maps(data_dict)
    plt.savefig(output_dir / 'gradient_comparison.png')
    plt.close()
    
    # 分析每种策略的奇偶层
    results = {}
    for name, data in data_dict.items():
        if data is not None:
            plt.figure(figsize=(15, 8))
            stats = analyze_odd_even_layers(data, title=name)
            plt.savefig(output_dir / f'odd_even_analysis_{name.lower().replace(" ", "_")}.png')
            plt.close()
            results[name] = stats
    
    # 生成分析报告
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write("Gradient Analysis Report\n")
        f.write("======================\n\n")
        
        for strategy, stats in results.items():
            f.write(f"\n{strategy}:\n")
            f.write("-" * (len(strategy) + 1) + "\n")
            f.write(f"Odd layers  - Mean: {stats['odd_mean']:.4f}, Std: {stats['odd_std']:.4f}\n")
            f.write(f"Even layers - Mean: {stats['even_mean']:.4f}, Std: {stats['even_std']:.4f}\n")
            f.write(f"Difference (Odd-Even): {(stats['odd_mean'] - stats['even_mean']):.4f}\n")

if __name__ == "__main__":
    main() 