#!/usr/bin/env python3
"""
专门测试von shell配置的完整链路
"""

import torch
import numpy as np
import warnings
from pathlib import Path
import sys

# 添加当前目录到路径
sys.path.append('.')

from utils import parse_cfg
from models import get_model
from datasets import get_dataset
from output_calculators import get_output_calculator
from loss_functions import get_loss_function
from ml_utils import Trainer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def test_von_shell_pipeline():
    """测试von shell完整链路"""
    print("=== Von Shell 完整链路测试 ===\n")
    
    # 设置数值稳定性
    torch.set_default_dtype(torch.float32)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 加载配置
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    print("✓ 配置文件加载成功")
    
    # 验证配置
    assert cfg["model_name"] == "von_shell", "模型名称必须是von_shell"
    assert cfg["dataset_name"] == "von_shell", "数据集名称必须是von_shell"
    assert cfg["output_calculator"] == "von_shell", "输出计算器必须是von_shell"
    print("✓ 配置验证通过")
    
    # 创建数据集
    print("\n--- 创建数据集 ---")
    dataset = get_dataset(cfg)
    print(f"✓ 数据集创建成功，类型: {type(dataset).__name__}")
    print(f"  数据点数量: {len(dataset)}")
    
    # 测试数据样本
    sample_input, sample_output = dataset[0]
    print(f"  输入形状: {sample_input.shape}")
    print(f"  输出形状: {sample_output.shape}")
    
    # 验证输出通道数
    assert sample_output.shape[0] == 16, f"输出通道数应该是16，实际是{sample_output.shape[0]}"
    print("✓ 输出通道数验证通过")
    
    # 检查数据是否包含NaN
    if torch.isnan(sample_input).any():
        print("✗ 输入数据包含NaN")
        return False
    if torch.isnan(sample_output).any():
        print("✗ 输出数据包含NaN")
        return False
    print("✓ 数据质量检查通过")
    
    # 创建模型
    print("\n--- 创建模型 ---")
    model = get_model(cfg)
    print(f"✓ 模型创建成功，类型: {type(model).__name__}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 检查模型参数
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            print(f"✗ 参数 {name} 包含NaN")
            return False
    print("✓ 模型参数检查通过")
    
    # 创建输出计算器
    print("\n--- 创建输出计算器 ---")
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)
    print(f"✓ 输出计算器创建成功，类型: {type(output_calculator).__name__}")
    
    # 创建损失函数
    print("\n--- 创建损失函数 ---")
    loss_fn = get_loss_function(cfg)
    print(f"✓ 损失函数创建成功")
    
    # 测试前向传播
    print("\n--- 测试前向传播 ---")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # 测试单个样本
        input_data = sample_input.to(device).unsqueeze(0)
        model_output = model(input_data)
        print(f"  模型输出形状: {model_output.shape}")
        
        # 验证模型输出
        assert model_output.shape[1] == 4, f"模型输出应该是4个参数，实际是{model_output.shape[1]}"
        print("✓ 模型输出形状验证通过")
        
        if torch.isnan(model_output).any():
            print("✗ 模型输出包含NaN")
            return False
        print("✓ 模型输出质量检查通过")
        
        # 测试输出计算器
        calculated_output, kwargs = output_calculator.output_from_model_out(model_output)
        print(f"  计算输出形状: {calculated_output.shape}")
        
        # 验证计算输出
        assert calculated_output.shape[1] == 16, f"计算输出应该是16个方向，实际是{calculated_output.shape[1]}"
        print("✓ 计算输出形状验证通过")
        
        if torch.isnan(calculated_output).any():
            print("✗ 计算输出包含NaN")
            return False
        print("✓ 计算输出质量检查通过")
        
        # 测试损失计算
        labels = sample_output.to(device).unsqueeze(0)
        loss = loss_fn(calculated_output, labels, **kwargs)
        print(f"  损失值: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("✗ 损失值异常")
            return False
        print("✓ 损失计算通过")
    
    # 测试训练步骤
    print("\n--- 测试训练步骤 ---")
    model.train()
    
    # 创建小批次数据
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    # 创建优化器
    train_cfg = cfg["train_cfg"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], eps=1e-8)
    
    # 测试一个训练步骤
    for batch_idx, (input_data, labels) in enumerate(dataloader):
        if batch_idx >= 1:  # 只测试第一个批次
            break
            
        input_data = input_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        model_output = model(input_data)
        calculated_output, kwargs = output_calculator.output_from_model_out(model_output)
        loss = loss_fn(calculated_output, labels, **kwargs)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        # 梯度裁剪
        if total_norm > 1.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        print(f"  批次大小: {input_data.shape[0]}")
        print(f"  损失值: {loss.item():.6f}")
        print(f"  梯度范数: {total_norm:.4f}")
        
        # 检查是否包含NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("✗ 损失值异常")
            return False
        
        # 检查模型参数
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                print(f"✗ 参数 {name} 包含异常值")
                return False
        
        print("✓ 训练步骤通过")
        break
    
    print("\n=== 所有测试通过！ ===")
    print("✓ Von Shell 链路完全正常")
    print("✓ 没有NaN/Inf问题")
    print("✓ 可以开始正式训练")
    
    return True


def main():
    """主函数"""
    try:
        success = test_von_shell_pipeline()
        if success:
            print("\n🎉 Von Shell 配置修复成功！")
            return True
        else:
            print("\n❌ Von Shell 配置仍有问题")
            return False
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 