"""
数值稳定性工具模块
用于解决训练过程中的NaN和数值不稳定问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional


class NumericalStabilizer:
    """数值稳定性管理器"""
    
    def __init__(self, eps: float = 1e-8, max_val: float = 1e6):
        self.eps = eps
        self.max_val = max_val
        
    def stabilize_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """稳定张量数值"""
        if torch.isnan(tensor).any():
            print(f"警告: {name} 包含 NaN，已替换为0")
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            
        if torch.isinf(tensor).any():
            print(f"警告: {name} 包含 Inf，已裁剪")
            tensor = torch.clamp(tensor, -self.max_val, self.max_val)
            
        return tensor
        
    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        """安全的对数运算"""
        return torch.log(torch.clamp(x, min=self.eps))
        
    def safe_sqrt(self, x: torch.Tensor) -> torch.Tensor:
        """安全的平方根运算"""
        return torch.sqrt(torch.clamp(x, min=self.eps))
        
    def safe_divide(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """安全的除法运算"""
        return numerator / torch.clamp(denominator, min=self.eps)
        
    def safe_exp(self, x: torch.Tensor) -> torch.Tensor:
        """安全的指数运算"""
        return torch.exp(torch.clamp(x, max=20))  # 防止exp爆炸


class ModelInitializer:
    """模型初始化器"""
    
    @staticmethod
    def xavier_uniform_scaled(tensor: torch.Tensor, gain: float = 0.1):
        """缩放的Xavier均匀初始化"""
        nn.init.xavier_uniform_(tensor, gain=gain)
        
    @staticmethod
    def normal_scaled(tensor: torch.Tensor, std: float = 0.01):
        """缩放的正态初始化"""
        nn.init.normal_(tensor, mean=0.0, std=std)
        
    @staticmethod
    def orthogonal_scaled(tensor: torch.Tensor, gain: float = 0.1):
        """缩放的正交初始化"""
        nn.init.orthogonal_(tensor, gain=gain)
        
    @classmethod
    def init_model_safe(cls, model: nn.Module, method: str = "xavier"):
        """安全初始化模型"""
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                if method == "xavier":
                    cls.xavier_uniform_scaled(param.data, gain=0.1)
                elif method == "normal":
                    cls.normal_scaled(param.data, std=0.01)
                elif method == "orthogonal":
                    cls.orthogonal_scaled(param.data, gain=0.1)
            else:
                nn.init.zeros_(param.data)
                
            print(f"初始化 {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")


class GradientProcessor:
    """梯度处理器"""
    
    def __init__(self, max_norm: float = 1.0, eps: float = 1e-8):
        self.max_norm = max_norm
        self.eps = eps
        
    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """检查梯度状态"""
        stats = {
            "total_norm": 0.0,
            "nan_count": 0,
            "inf_count": 0,
            "zero_count": 0,
            "param_count": 0
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats["param_count"] += 1
                
                # 检查NaN
                if torch.isnan(grad).any():
                    stats["nan_count"] += 1
                    print(f"梯度 {name} 包含 NaN")
                    
                # 检查Inf
                if torch.isinf(grad).any():
                    stats["inf_count"] += 1
                    print(f"梯度 {name} 包含 Inf")
                    
                # 检查零梯度
                if grad.norm() < self.eps:
                    stats["zero_count"] += 1
                    print(f"梯度 {name} 接近零")
                    
                # 计算范数
                param_norm = grad.norm(2)
                stats["total_norm"] += param_norm.item() ** 2
                
        stats["total_norm"] = stats["total_norm"] ** 0.5
        return stats
        
    def fix_gradients(self, model: nn.Module) -> bool:
        """修复异常梯度"""
        fixed_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # 修复NaN
                if torch.isnan(grad).any():
                    param.grad.data = torch.where(torch.isnan(grad), 
                                                torch.zeros_like(grad), grad)
                    fixed_count += 1
                    
                # 修复Inf
                if torch.isinf(grad).any():
                    param.grad.data = torch.clamp(grad, -100, 100)
                    fixed_count += 1
                    
        return fixed_count > 0
        
    def clip_gradients(self, model: nn.Module) -> float:
        """裁剪梯度"""
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        return total_norm.item()


class LossStabilizer:
    """损失稳定器"""
    
    def __init__(self, eps: float = 1e-8, max_loss: float = 1000.0):
        self.eps = eps
        self.max_loss = max_loss
        
    def stabilize_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """稳定的MSE损失"""
        diff = pred - target
        # 使用Huber损失的思想，对大的差值进行平滑
        abs_diff = torch.abs(diff)
        square_loss = 0.5 * diff * diff
        linear_loss = abs_diff - 0.5
        
        # 当差值小于1时使用平方损失，否则使用线性损失
        loss = torch.where(abs_diff < 1.0, square_loss, linear_loss)
        return loss.mean()
        
    def stabilize_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """稳定损失值"""
        if torch.isnan(loss):
            print("损失为NaN，返回默认损失")
            return torch.tensor(1.0, device=loss.device, requires_grad=True)
            
        if torch.isinf(loss):
            print("损失为Inf，返回最大损失")
            return torch.tensor(self.max_loss, device=loss.device, requires_grad=True)
            
        # 裁剪损失值
        return torch.clamp(loss, max=self.max_loss)


class ActivationStabilizer:
    """激活函数稳定器"""
    
    @staticmethod
    def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """稳定的softmax"""
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
        
    @staticmethod
    def stable_relu(x: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
        """稳定的ReLU"""
        return torch.clamp(F.relu(x), max=max_val)
        
    @staticmethod
    def stable_tanh(x: torch.Tensor) -> torch.Tensor:
        """稳定的tanh"""
        return torch.tanh(torch.clamp(x, -10, 10))
        
    @staticmethod
    def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
        """稳定的sigmoid"""
        return torch.sigmoid(torch.clamp(x, -10, 10))


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history = []
        self.nan_count = 0
        self.inf_count = 0
        
    def update(self, loss: float, has_nan: bool = False, has_inf: bool = False):
        """更新监控状态"""
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            
        if has_nan:
            self.nan_count += 1
        if has_inf:
            self.inf_count += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        if not self.loss_history:
            return {}
            
        losses = np.array(self.loss_history)
        return {
            "mean_loss": losses.mean(),
            "std_loss": losses.std(),
            "min_loss": losses.min(),
            "max_loss": losses.max(),
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "loss_trend": losses[-10:].mean() - losses[:10].mean() if len(losses) >= 20 else 0
        }
        
    def is_training_stable(self) -> bool:
        """检查训练是否稳定"""
        if len(self.loss_history) < 20:
            return True
            
        recent_losses = self.loss_history[-10:]
        return all(loss < 1000 for loss in recent_losses) and self.nan_count == 0


# 工具函数
def create_stable_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-6) -> torch.optim.Optimizer:
    """创建稳定的优化器"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )


def create_stable_scheduler(optimizer: torch.optim.Optimizer, 
                          factor: float = 0.8, 
                          patience: int = 10) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """创建稳定的学习率调度器"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        verbose=True,
        threshold=1e-6,
        threshold_mode='rel',
        cooldown=5,
        min_lr=1e-7,
        eps=1e-8
    )


def diagnose_model_health(model: nn.Module) -> Dict[str, Any]:
    """诊断模型健康状况"""
    health_report = {
        "total_params": 0,
        "nan_params": 0,
        "inf_params": 0,
        "zero_params": 0,
        "param_stats": {}
    }
    
    for name, param in model.named_parameters():
        health_report["total_params"] += param.numel()
        
        if torch.isnan(param).any():
            health_report["nan_params"] += torch.isnan(param).sum().item()
            
        if torch.isinf(param).any():
            health_report["inf_params"] += torch.isinf(param).sum().item()
            
        if param.abs().max() < 1e-8:
            health_report["zero_params"] += param.numel()
            
        health_report["param_stats"][name] = {
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "min": param.data.min().item(),
            "max": param.data.max().item()
        }
        
    return health_report
