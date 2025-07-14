"""
NaN问题修复前后的对比测试
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class OldModel(nn.Module):
    """模拟原始可能产生NaN的模型"""
    def __init__(self, input_size=20, hidden_dim=256, n_layers=5):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 45))
        self.mlp = nn.Sequential(*layers)
        
        # 使用可能产生NaN的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.0)  # 较大的标准差
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.mlp(x)


def test_model_stability(model, name: str, test_rounds: int = 100) -> Dict:
    """测试模型的稳定性"""
    nan_count = 0
    inf_count = 0
    loss_values = []
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 较大的学习率
    
    for round_idx in range(test_rounds):
        # 生成测试数据 - 修正维度
        if hasattr(model, 'Lpos'):
            # 对于稳定模型，使用3维输入（xyz坐标）
            x = torch.randn(32, 3) * 0.5
        else:
            # 对于原始模型，使用20维输入
            x = torch.randn(32, 20) * 2.0
        
        y = torch.randn(32, 45) * 0.1
        
        optimizer.zero_grad()
        
        try:
            output = model(x)
            
            # 检查输出
            if torch.isnan(output).any():
                nan_count += 1
                continue
                
            if torch.isinf(output).any():
                inf_count += 1
                continue
            
            # 计算损失
            loss = criterion(output, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue
                
            loss.backward()
            
            # 检查梯度
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                nan_count += 1
                continue
                
            optimizer.step()
            loss_values.append(loss.item())
            
        except Exception as e:
            print(f"{name} 第{round_idx}轮出错: {e}")
            nan_count += 1
    
    return {
        "name": name,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "success_rate": (test_rounds - nan_count - inf_count) / test_rounds,
        "avg_loss": np.mean(loss_values) if loss_values else float('inf'),
        "loss_std": np.std(loss_values) if loss_values else float('inf')
    }


def run_comparison():
    """运行对比测试"""
    print("=== NaN问题修复前后对比测试 ===\\n")
    
    # 测试原始模型
    print("测试原始模型（容易产生NaN）...")
    old_model = OldModel()
    old_results = test_model_stability(old_model, "原始模型")
    
    # 测试改进模型
    print("测试改进模型（数值稳定）...")
    from stable_models import StableFod_NeSH
    stable_model = StableFod_NeSH(lpos=10, hidden_dim=256, n_layers=5, max_freq=5.0, gaussian=False)
    stable_results = test_model_stability(stable_model, "改进模型")
    
    # 打印结果
    print("\\n=== 测试结果对比 ===")
    print(f"{'指标':<15} {'原始模型':<15} {'改进模型':<15} {'改进效果':<15}")
    print("-" * 60)
    
    # NaN次数
    print(f"{'NaN次数':<15} {old_results['nan_count']:<15} {stable_results['nan_count']:<15} {old_results['nan_count'] - stable_results['nan_count']:+d}")
    
    # 成功率
    print(f"{'成功率':<15} {old_results['success_rate']:<15.2%} {stable_results['success_rate']:<15.2%} {stable_results['success_rate'] - old_results['success_rate']:+.2%}")
    
    # 平均损失
    old_loss = old_results['avg_loss'] if old_results['avg_loss'] != float('inf') else 999
    stable_loss = stable_results['avg_loss'] if stable_results['avg_loss'] != float('inf') else 999
    print(f"{'平均损失':<15} {old_loss:<15.6f} {stable_loss:<15.6f} {stable_loss - old_loss:+.6f}")
    
    # 损失标准差
    old_std = old_results['loss_std'] if old_results['loss_std'] != float('inf') else 999
    stable_std = stable_results['loss_std'] if stable_results['loss_std'] != float('inf') else 999
    print(f"{'损失标准差':<15} {old_std:<15.6f} {stable_std:<15.6f} {stable_std - old_std:+.6f}")
    
    print("\\n=== 结论 ===")
    if stable_results['success_rate'] > old_results['success_rate']:
        print("✅ 改进模型显著提高了训练稳定性")
    else:
        print("❌ 改进效果不明显，需要进一步调整")
    
    if stable_results['nan_count'] < old_results['nan_count']:
        print("✅ 成功减少了NaN问题")
    else:
        print("❌ NaN问题仍然存在")
    
    print("\\n=== 建议 ===")
    print("1. 使用stable_models.py中的稳定模型")
    print("2. 采用更小的学习率（1e-4而不是1e-3）")
    print("3. 添加梯度裁剪和权重衰减")
    print("4. 使用improved_trainer.py进行训练")
    print("5. 监控训练过程，及时发现异常")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_comparison()
