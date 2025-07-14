"""
使用稳定训练组件的示例
解决NaN问题的完整解决方案
"""

import torch
import numpy as np
from pathlib import Path
import yaml

# 导入稳定的组件
from improved_trainer import ImprovedTrainer
from stable_models import StableFod_NeSH, StableVon_G
from numerical_stability import (
    NumericalStabilizer, ModelInitializer, GradientProcessor,
    LossStabilizer, create_stable_optimizer, create_stable_scheduler,
    diagnose_model_health
)

# 导入原有组件
from datasets import get_dataset
from output_calculators import get_output_calculator
from loss_functions import get_loss_function
from torch.utils.data import DataLoader


def load_stable_config(config_path: str) -> dict:
    """加载稳定训练配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_stable_model(cfg: dict) -> torch.nn.Module:
    """创建数值稳定的模型"""
    model_name = cfg["model_name"]
    train_cfg = cfg["train_cfg"]
    
    if model_name == "stable_fod_nesh":
        model = StableFod_NeSH(
            l_max=train_cfg["lmax"],
            lpos=train_cfg["lpos"],
            hidden_dim=train_cfg["hidden_dim"],
            n_layers=train_cfg["n_layers"],
            sigma=train_cfg.get("sigma"),
            gaussian=train_cfg.get("gaussian", True),
            dropout_rate=train_cfg.get("dropout_rate", 0.1),
            max_freq=train_cfg.get("max_freq", 5.0)
        )
    elif model_name == "stable_von_g":
        model = StableVon_G(
            l_max=train_cfg["lmax"],
            lpos=train_cfg["lpos"],
            hidden_dim=train_cfg["hidden_dim"],
            n_layers=train_cfg["n_layers"],
            sigma=train_cfg.get("sigma"),
            gaussian=train_cfg.get("gaussian", True),
            dropout_rate=train_cfg.get("dropout_rate", 0.1),
            max_freq=train_cfg.get("max_freq", 5.0)
        )
    else:
        raise ValueError(f"未知的模型类型: {model_name}")
    
    # 使用稳定的权重初始化
    init_method = cfg.get("model_init", {}).get("method", "xavier")
    ModelInitializer.init_model_safe(model, method=init_method)
    
    return model


def create_stable_dataloader(cfg: dict) -> tuple:
    """创建稳定的数据加载器"""
    dataset = get_dataset(cfg)
    
    # 数据预处理
    if cfg.get("data_processing", {}).get("normalize_input", False):
        # 在这里添加输入标准化逻辑
        pass
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train_cfg"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True  # 提高GPU传输效率
    )
    
    return dataset, dataloader


def run_stable_training():
    """运行稳定训练"""
    # 加载配置
    cfg = load_stable_config("configs/stable_config.yaml")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建稳定的模型
    model = create_stable_model(cfg)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 诊断模型健康状况
    health_report = diagnose_model_health(model)
    print(f"模型健康诊断: {health_report}")
    
    # 创建数据加载器
    dataset, dataloader = create_stable_dataloader(cfg)
    
    # 创建输出计算器
    output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)
    
    # 创建损失函数
    loss_fn = get_loss_function(cfg)
    
    # 创建稳定的优化器
    optimizer = create_stable_optimizer(
        model, 
        lr=cfg["train_cfg"]["lr"],
        weight_decay=cfg["train_cfg"].get("weight_decay", 1e-6)
    )
    
    # 创建学习率调度器
    scheduler = create_stable_scheduler(optimizer) if cfg.get("stability", {}).get("scheduler") else None
    
    # 创建改进的训练器
    trainer = ImprovedTrainer(
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=cfg["train_cfg"]["epochs"],
        l_max=cfg["train_cfg"]["lmax"],
        data_shape=(cfg["width"], cfg["height"], cfg["depth"]),
        output_calculator=output_calculator,
        scheduler=scheduler,
        patience=cfg.get("stability", {}).get("patience", 15),
        init_lr=cfg["train_cfg"]["lr"],
        warmup_epochs=cfg.get("stability", {}).get("warmup_epochs", 5),
        max_grad_norm=cfg.get("stability", {}).get("max_grad_norm", 1.0),
        weight_decay=cfg["train_cfg"].get("weight_decay", 1e-6)
    )
    
    # 开始训练
    print("开始稳定训练...")
    avg_loss = trainer.train()
    
    # 保存最终模型
    torch.save(model.state_dict(), "final_stable_model.pth")
    
    # 最终健康诊断
    final_health = diagnose_model_health(model)
    print(f"训练后模型健康状况: {final_health}")
    
    return model, avg_loss


def debug_nan_issues():
    """调试NaN问题的工具函数"""
    print("=== NaN问题调试工具 ===")
    
    # 加载配置
    cfg = load_stable_config("configs/stable_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = create_stable_model(cfg)
    model.to(device)
    
    # 创建数据
    dataset, dataloader = create_stable_dataloader(cfg)
    
    # 创建工具
    stabilizer = NumericalStabilizer()
    grad_processor = GradientProcessor()
    
    # 测试一个批次
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        print(f"\\n批次 {i}:")
        print(f"输入统计: min={inputs.min():.6f}, max={inputs.max():.6f}, mean={inputs.mean():.6f}")
        print(f"标签统计: min={labels.min():.6f}, max={labels.max():.6f}, mean={labels.mean():.6f}")
        
        # 前向传播
        outputs = model(inputs)
        print(f"输出统计: min={outputs.min():.6f}, max={outputs.max():.6f}, mean={outputs.mean():.6f}")
        
        # 检查NaN
        if torch.isnan(outputs).any():
            print("❌ 输出包含NaN！")
            # 在这里添加详细的调试信息
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"参数 {name} 包含NaN")
        else:
            print("✅ 输出正常")
            
        # 只测试几个批次
        if i >= 3:
            break
            
    print("\\n=== 调试完成 ===")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 选择运行模式
    mode = "train"  # 或 "debug"
    
    if mode == "train":
        model, losses = run_stable_training()
        print("训练完成！")
    elif mode == "debug":
        debug_nan_issues()
    else:
        print("未知模式，请选择 'train' 或 'debug'")
