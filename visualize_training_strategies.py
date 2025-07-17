import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import stats
from matplotlib.colors import Normalize

def load_results():
    """加载训练结果"""
    results_dir = Path("results/training_strategies")
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Summary file not found. Please run training first.")
    return pd.read_csv(summary_path)

def plot_training_metrics(df):
    """绘制训练指标对比图"""
    plt.figure(figsize=(15, 10))
    
    # 训练时间对比
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x="config_name", y="training_time")
    plt.title("Training Time Comparison")
    plt.xticks(rotation=45)
    
    # 预测值分布
    plt.subplot(2, 2, 2)
    metrics = ["mean", "std"]
    df_melted = df.melt(id_vars="config_name", value_vars=metrics)
    sns.barplot(data=df_melted, x="config_name", y="value", hue="variable")
    plt.title("Prediction Statistics")
    plt.xticks(rotation=45)
    
    # 非零值比例
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x="config_name", y="non_zero")
    plt.title("Non-zero Ratio")
    plt.xticks(rotation=45)
    
    # 值范围
    plt.subplot(2, 2, 4)
    df_range = df.melt(id_vars="config_name", value_vars=["min", "max"])
    sns.barplot(data=df_range, x="config_name", y="value", hue="variable")
    plt.title("Value Range")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/training_strategies/metrics_comparison.png")
    plt.close()

def compare_predictions():
    """比较不同策略的预测结果"""
    slice_id = 45  # 选择要比较的slice
    
    # 加载所有预测结果
    predictions = {}
    for config in ["odd_even_split", "random_70", "random_50", "random_30"]:
        pred_path = Path(f"runs/training_strategies/{config}/coeffs_test.nii.gz")
        if pred_path.exists():
            img = nib.load(str(pred_path))
            data = img.get_fdata()
            predictions[config] = data[:, :, slice_id, 0]  # 取第一个通道
    
    if not predictions:
        print("No prediction files found")
        return
        
    # 创建对比图
    n_configs = len(predictions)
    fig, axes = plt.subplots(2, n_configs, figsize=(5*n_configs, 10))
    
    # 第一行：预测结果
    for i, (config, pred) in enumerate(predictions.items()):
        im = axes[0, i].imshow(pred, cmap="gray")
        axes[0, i].set_title(f"{config}\nPrediction")
        plt.colorbar(im, ax=axes[0, i])
        
    # 第二行：与基准(odd_even_split)的差异
    base_pred = predictions.get("odd_even_split")
    if base_pred is not None:
        for i, (config, pred) in enumerate(predictions.items()):
            if config != "odd_even_split":
                diff = pred - base_pred
                im = axes[1, i].imshow(diff, cmap="RdBu", norm=Normalize(vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff))))
                axes[1, i].set_title(f"{config}\nvs odd_even_split")
                plt.colorbar(im, ax=axes[1, i])
            else:
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/training_strategies/predictions_comparison.png")
    plt.close()

def calculate_correlations():
    """计算不同策略预测结果之间的相关性"""
    slice_id = 45
    
    # 加载预测结果
    predictions = {}
    for config in ["odd_even_split", "random_70", "random_50", "random_30"]:
        pred_path = Path(f"runs/training_strategies/{config}/coeffs_test.nii.gz")
        if pred_path.exists():
            img = nib.load(str(pred_path))
            data = img.get_fdata()
            predictions[config] = data[:, :, slice_id, 0].flatten()
    
    if not predictions:
        print("No prediction files found")
        return
        
    # 计算相关系数
    configs = list(predictions.keys())
    corr_matrix = np.zeros((len(configs), len(configs)))
    
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs):
            corr, _ = stats.pearsonr(predictions[config1], predictions[config2])
            corr_matrix[i, j] = corr
    
    # 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", xticklabels=configs, yticklabels=configs)
    plt.title("Prediction Correlations")
    plt.tight_layout()
    plt.savefig("results/training_strategies/correlations.png")
    plt.close()

def main():
    # 加载结果
    df = load_results()
    
    # 绘制训练指标对比图
    plot_training_metrics(df)
    
    # 比较预测结果
    compare_predictions()
    
    # 计算相关性
    calculate_correlations()
    
    print("Visualization completed. Please check results/training_strategies/ directory.")

if __name__ == "__main__":
    main() 