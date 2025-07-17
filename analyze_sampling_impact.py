import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats

def load_gradient_data(file_path):
    """加载梯度数据文件"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        if len(data.shape) > 3:
            # 计算每个体素的梯度幅值
            data = np.sqrt(np.sum(data**2, axis=-1))
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_metrics(pred_data, ref_data):
    """计算预测质量指标"""
    # 确保数据形状一致
    assert pred_data.shape == ref_data.shape
    
    # 计算各种指标
    mse = np.mean((pred_data - ref_data) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_data - ref_data))
    
    # 计算相关系数
    if np.std(pred_data) > 0 and np.std(ref_data) > 0:
        correlation = stats.pearsonr(pred_data.flatten(), ref_data.flatten())[0]
    else:
        correlation = 0
    
    # 计算结构相似性
    def ssim(y_true, y_pred):
        mu_true = np.mean(y_true)
        mu_pred = np.mean(y_pred)
        sigma_true = np.std(y_true)
        sigma_pred = np.std(y_pred)
        sigma_true_pred = np.mean((y_true - mu_true) * (y_pred - mu_pred))
        
        c1 = (0.01 * np.max(y_true)) ** 2
        c2 = (0.03 * np.max(y_true)) ** 2
        
        if (mu_true**2 + mu_pred**2 + c1) * (sigma_true**2 + sigma_pred**2 + c2) > 0:
            ssim = ((2 * mu_true * mu_pred + c1) * (2 * sigma_true_pred + c2)) / \
                   ((mu_true**2 + mu_pred**2 + c1) * (sigma_true**2 + sigma_pred**2 + c2))
        else:
            ssim = 0
        return ssim
    
    similarity = ssim(ref_data, pred_data)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'similarity': similarity
    }

def analyze_layer_wise(pred_data, ref_data):
    """分析每一层的预测质量"""
    n_layers = pred_data.shape[2]
    metrics_per_layer = []
    
    for i in range(n_layers):
        metrics = calculate_metrics(pred_data[:,:,i], ref_data[:,:,i])
        metrics_per_layer.append(metrics)
    
    return metrics_per_layer

def plot_metrics_comparison(strategies, metrics_dict, output_dir):
    """绘制不同策略下的指标对比图"""
    plt.figure(figsize=(15, 10))
    metrics_names = ['RMSE', 'MAE', 'Correlation', 'Similarity']
    metrics_keys = ['rmse', 'mae', 'correlation', 'similarity']
    
    x = np.arange(len(strategies))
    width = 0.2
    
    for idx, (name, key) in enumerate(zip(metrics_names, metrics_keys)):
        plt.subplot(2, 2, idx+1)
        values = [metrics_dict[strategy][key] for strategy in strategies]
        plt.bar(x, values, width=width)
        plt.title(f'{name} Comparison')
        plt.xticks(x, strategies, rotation=45)
        plt.ylabel(name)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png')
    plt.close()

def plot_layer_metrics(layer_metrics_dict, strategies, output_dir):
    """绘制层级指标分析图"""
    n_layers = len(list(layer_metrics_dict.values())[0])
    metrics_names = ['RMSE', 'MAE', 'Correlation', 'Similarity']
    metrics_keys = ['rmse', 'mae', 'correlation', 'similarity']
    
    plt.figure(figsize=(15, 10))
    for idx, (name, key) in enumerate(zip(metrics_names, metrics_keys)):
        plt.subplot(2, 2, idx+1)
        
        for strategy in strategies:
            layer_metrics = layer_metrics_dict[strategy]
            values = [metrics[key] for metrics in layer_metrics]
            plt.plot(range(n_layers), values, label=strategy)
            
        plt.title(f'Layer-wise {name}')
        plt.xlabel('Layer Index')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_wise_metrics.png')
    plt.close()

def generate_markdown_report(metrics_dict, strategies, output_dir):
    """生成Markdown格式的分析报告"""
    report = "# 不同训练策略的预测质量分析\n\n"
    
    # 总体趋势分析
    report += "## 1. 总体性能分析\n\n"
    report += "### 1.1 各项指标对比\n\n"
    report += "| 策略 | RMSE | MAE | 相关系数 | 结构相似性 |\n"
    report += "|------|------|-----|----------|------------|\n"
    
    for strategy in strategies:
        metrics = metrics_dict[strategy]
        report += f"| {strategy} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
        report += f"{metrics['correlation']:.4f} | {metrics['similarity']:.4f} |\n"
    
    # 关键发现
    report += "\n## 2. 关键发现\n\n"
    
    # 计算相对变化
    base_metrics = metrics_dict['Full Training']
    for metric in ['rmse', 'mae', 'correlation', 'similarity']:
        report += f"### 2.1 {metric.upper()} 分析\n\n"
        report += "- 相对于全量训练的变化：\n"
        for strategy in strategies:
            if strategy != 'Full Training':
                relative_change = (metrics_dict[strategy][metric] - base_metrics[metric]) / base_metrics[metric] * 100
                report += f"  * {strategy}：变化 {relative_change:.2f}%\n"
        report += "\n"
    
    # 建议
    report += "## 3. 结论与建议\n\n"
    
    # 找出最佳策略
    best_strategy = 'Full Training'
    best_score = 0
    for strategy, metrics in metrics_dict.items():
        # 综合得分（可以根据需要调整权重）
        score = (metrics['correlation'] + metrics['similarity']) / 2 - (metrics['rmse'] + metrics['mae']) / (2 * max(metrics_dict['Full Training']['rmse'], metrics_dict['Full Training']['mae']))
        if score > best_score:
            best_score = score
            best_strategy = strategy
    
    report += f"1. 最佳策略推荐：{best_strategy}\n\n"
    report += "2. 性能分析：\n"
    for strategy in strategies:
        report += f"   - {strategy}：\n"
        metrics = metrics_dict[strategy]
        report += f"     * RMSE: {metrics['rmse']:.4f}\n"
        report += f"     * 相关系数: {metrics['correlation']:.4f}\n"
        report += f"     * 结构相似性: {metrics['similarity']:.4f}\n"
    
    report += "\n3. 建议：\n"
    report += "   - 计算资源与性能的权衡\n"
    report += "   - 特定场景的适用性分析\n"
    report += "   - 进一步优化的可能方向\n"
    
    # 保存报告
    with open(output_dir / 'strategy_analysis_report.md', 'w') as f:
        f.write(report)

def main():
    # 设置数据路径和策略
    base_path = Path('runs/mwu100408_250716')
    strategies = {
        'Full Training': 'grads_test.nii.gz',
        '70% Sampling': 'grads_colfg8vl.nii.gz',
        'Odd-Even Split': 'grads_tcqivmvu.nii.gz'
    }
    
    # 创建输出目录
    output_dir = Path('strategy_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # 加载参考数据（全量训练）
    ref_data = load_gradient_data(str(base_path / strategies['Full Training']))
    
    # 分析不同策略的结果
    metrics_dict = {}
    layer_metrics_dict = {}
    
    for strategy_name, file_name in strategies.items():
        data = load_gradient_data(str(base_path / file_name))
        if data is not None:
            metrics = calculate_metrics(data, ref_data)
            metrics_dict[strategy_name] = metrics
            
            # 计算层级指标
            layer_metrics = analyze_layer_wise(data, ref_data)
            layer_metrics_dict[strategy_name] = layer_metrics
    
    # 生成可视化结果
    plot_metrics_comparison(list(strategies.keys()), metrics_dict, output_dir)
    plot_layer_metrics(layer_metrics_dict, list(strategies.keys()), output_dir)
    
    # 生成报告
    generate_markdown_report(metrics_dict, list(strategies.keys()), output_dir)

if __name__ == "__main__":
    main() 