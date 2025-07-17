import os
import yaml
from pathlib import Path
import subprocess
import time
import pandas as pd
import numpy as np
import nibabel as nib
from datetime import datetime

def load_config(config_name):
    """加载指定配置"""
    with open("configs/training_strategies.yaml", "r") as f:
        configs = yaml.safe_load(f)
    return configs[config_name]

def run_training(config_name):
    """运行指定配置的训练"""
    config = load_config(config_name)
    
    # 保存当前配置到临时文件
    temp_config_path = f"configs/temp_{config_name}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    # 运行训练
    cmd = f"python main.py --config {temp_config_path}"
    start_time = time.time()
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()
    
    # 删除临时配置文件
    os.remove(temp_config_path)
    
    training_time = end_time - start_time
    return {
        "config_name": config_name,
        "training_time": training_time,
        "stdout": stdout.decode(),
        "stderr": stderr.decode()
    }

def evaluate_results(config_name):
    """评估训练结果"""
    config = load_config(config_name)
    output_dir = Path(config["paths"]["output"])
    
    # 加载预测结果
    pred_path = output_dir / "coeffs_test.nii.gz"
    if not pred_path.exists():
        return None
        
    pred_img = nib.load(str(pred_path))
    pred_data = pred_img.get_fdata()
    
    # 计算基本统计信息
    stats = {
        "mean": float(np.mean(pred_data)),
        "std": float(np.std(pred_data)),
        "min": float(np.min(pred_data)),
        "max": float(np.max(pred_data)),
        "non_zero": float(np.count_nonzero(pred_data) / pred_data.size)
    }
    
    return stats

def main():
    # 创建结果目录
    results_dir = Path("results/training_strategies")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 所有配置
    configs = ["odd_even_split", "random_70", "random_50", "random_30"]
    
    # 运行所有训练策略
    results = []
    for config_name in configs:
        print(f"\n=== Running {config_name} ===")
        
        # 运行训练
        training_result = run_training(config_name)
        
        # 评估结果
        eval_stats = evaluate_results(config_name)
        
        if eval_stats:
            result = {
                "config_name": config_name,
                "training_time": training_result["training_time"],
                **eval_stats
            }
            results.append(result)
            
        # 保存详细日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = results_dir / f"{config_name}_{timestamp}.log"
        with open(log_path, "w") as f:
            f.write(f"=== Training Log ===\n")
            f.write(f"Config: {config_name}\n")
            f.write(f"Training Time: {training_result['training_time']:.2f}s\n")
            f.write("\n=== stdout ===\n")
            f.write(training_result["stdout"])
            f.write("\n=== stderr ===\n")
            f.write(training_result["stderr"])
    
    # 保存汇总结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "summary.csv", index=False)
        print("\n=== Summary ===")
        print(df)

if __name__ == "__main__":
    main() 