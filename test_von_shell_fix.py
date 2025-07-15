#!/usr/bin/env python3
"""
测试von shell配置的修复效果
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


def test_data_loading():
    """测试数据加载"""
    print("=== 测试数据加载 ===")
    
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    
    try:
        dataset = get_dataset(cfg)
        print(f"✓ 数据集加载成功，数据点数量: {len(dataset)}")
        
        # 测试几个样本
        for i in range(min(5, len(dataset))):
            input_data, output_data = dataset[i]
            print(f"  样本 {i}: 输入形状 {input_data.shape}, 输出形状 {output_data.shape}")
            
            # 检查数据是否包含NaN
            if torch.isnan(input_data).any():
                print(f"  ✗ 样本 {i} 输入包含NaN")
            else:
                print(f"  ✓ 样本 {i} 输入正常")
                
            if torch.isnan(output_data).any():
                print(f"  ✗ 样本 {i} 输出包含NaN")
            else:
                print(f"  ✓ 样本 {i} 输出正常")
        
        return dataset
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        return None


def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    
    try:
        model = get_model(cfg)
        print(f"✓ 模型创建成功")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 检查模型参数
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                print(f"  ✗ 参数 {name} 包含NaN")
            else:
                print(f"  ✓ 参数 {name} 正常")
        
        return model
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return None


def test_forward_pass(model, dataset):
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    
    if model is None or dataset is None:
        print("✗ 模型或数据集为空，跳过测试")
        return False
    
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    try:
        # 测试单个样本
        input_data, output_data = dataset[0]
        input_data = input_data.to(device).unsqueeze(0)  # 添加batch维度
        
        with torch.no_grad():
            model_output = model(input_data)
            print(f"✓ 前向传播成功")
            print(f"  输入形状: {input_data.shape}")
            print(f"  输出形状: {model_output.shape}")
            
            if torch.isnan(model_output).any():
                print("  ✗ 模型输出包含NaN")
                return False
            else:
                print("  ✓ 模型输出正常")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False


def test_output_calculator(model, dataset):
    """测试输出计算器"""
    print("\n=== 测试输出计算器 ===")
    
    if model is None or dataset is None:
        print("✗ 模型或数据集为空，跳过测试")
        return False
    
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    
    try:
        output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)
        print("✓ 输出计算器创建成功")
        
        # 测试输出计算
        input_data, output_data = dataset[0]
        input_data = input_data.to(device).unsqueeze(0)
        
        with torch.no_grad():
            model_output = model(input_data)
            calculated_output, kwargs = output_calculator.output_from_model_out(model_output)
            
            print(f"✓ 输出计算成功")
            print(f"  计算输出形状: {calculated_output.shape}")
            
            if torch.isnan(calculated_output).any():
                print("  ✗ 计算输出包含NaN")
                return False
            else:
                print("  ✓ 计算输出正常")
        
        return True
    except Exception as e:
        print(f"✗ 输出计算器测试失败: {e}")
        return False


def test_loss_function(model, dataset):
    """测试损失函数"""
    print("\n=== 测试损失函数 ===")
    
    if model is None or dataset is None:
        print("✗ 模型或数据集为空，跳过测试")
        return False
    
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    
    try:
        loss_fn = get_loss_function(cfg)
        output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)
        print("✓ 损失函数创建成功")
        
        # 测试损失计算
        input_data, labels = dataset[0]
        input_data = input_data.to(device).unsqueeze(0)
        labels = labels.to(device).unsqueeze(0)
        
        with torch.no_grad():
            model_output = model(input_data)
            output, kwargs = output_calculator.output_from_model_out(model_output)
            loss = loss_fn(output, labels, **kwargs)
            
            print(f"✓ 损失计算成功")
            print(f"  损失值: {loss.item():.6f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("  ✗ 损失值异常")
                return False
            else:
                print("  ✓ 损失值正常")
        
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False


def test_training_step(model, dataset):
    """测试训练步骤"""
    print("\n=== 测试训练步骤 ===")
    
    if model is None or dataset is None:
        print("✗ 模型或数据集为空，跳过测试")
        return False
    
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    
    try:
        # 创建小批次数据
        dataloader = DataLoader(
            dataset,
            batch_size=4,  # 使用小批次
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            drop_last=True,
        )
        
        # 创建优化器和损失函数
        train_cfg = cfg["train_cfg"]
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], eps=1e-8)
        loss_fn = get_loss_function(cfg)
        output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)
        
        model.train()
        model.to(device)
        
        # 测试一个训练步骤
        for batch_idx, (input_data, labels) in enumerate(dataloader):
            if batch_idx >= 1:  # 只测试第一个批次
                break
                
            input_data = input_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            model_output = model(input_data)
            output, kwargs = output_calculator.output_from_model_out(model_output)
            loss = loss_fn(output, labels, **kwargs)
            
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
            
            print(f"✓ 训练步骤成功")
            print(f"  批次大小: {input_data.shape[0]}")
            print(f"  损失值: {loss.item():.6f}")
            print(f"  梯度范数: {total_norm:.4f}")
            
            # 检查是否包含NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("  ✗ 损失值异常")
                return False
            else:
                print("  ✓ 损失值正常")
            
            # 检查模型参数
            has_nan = False
            for name, param in model.named_parameters():
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    print(f"  ✗ 参数 {name} 包含异常值")
                    has_nan = True
            
            if not has_nan:
                print("  ✓ 模型参数正常")
                return True
            else:
                return False
        
        return True
    except Exception as e:
        print(f"✗ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始von shell配置测试...\n")
    
    # 设置数值稳定性
    torch.set_default_dtype(torch.float32)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 运行所有测试
    tests = [
        ("数据加载", test_data_loading),
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass),
        ("输出计算器", test_output_calculator),
        ("损失函数", test_loss_function),
        ("训练步骤", test_training_step),
    ]
    
    results = {}
    dataset = None
    model = None
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            if test_name == "数据加载":
                dataset = test_func()
                results[test_name] = dataset is not None
            elif test_name == "模型创建":
                model = test_func()
                results[test_name] = model is not None
            elif test_name == "前向传播":
                results[test_name] = test_func(model, dataset)
            elif test_name == "输出计算器":
                results[test_name] = test_func(model, dataset)
            elif test_name == "损失函数":
                results[test_name] = test_func(model, dataset)
            elif test_name == "训练步骤":
                results[test_name] = test_func(model, dataset)
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            results[test_name] = False
    
    # 打印测试结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print('='*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！von shell配置修复成功。")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步调试。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 