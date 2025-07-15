"""
数值稳定性工具模块
用于解决训练过程中的NaN和数值不稳定问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings


class NumericalStabilizer:
    """数值稳定性监控器和修复器"""
    
    def __init__(self, 
                 max_loss_threshold: float = 100.0,
                 max_grad_norm: float = 0.5,
                 eps: float = 1e-8,
                 enable_logging: bool = True):
        self.max_loss_threshold = max_loss_threshold
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.enable_logging = enable_logging
        self.stats = {
            'nan_detected': 0,
            'inf_detected': 0,
            'loss_exploded': 0,
            'grad_exploded': 0,
            'params_corrected': 0
        }
    
    def log_warning(self, message: str):
        """记录警告信息"""
        if self.enable_logging:
            print(f"数值稳定性警告: {message}")
    
    def check_and_fix_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """检查并修复张量中的数值问题"""
        if tensor is None:
            return tensor
            
        original_shape = tensor.shape
        
        # 检查NaN
        if torch.isnan(tensor).any():
            self.stats['nan_detected'] += 1
            self.log_warning(f"{name} 包含 NaN 值，进行修正")
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        
        # 检查Inf
        if torch.isinf(tensor).any():
            self.stats['inf_detected'] += 1
            self.log_warning(f"{name} 包含 Inf 值，进行修正")
            tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        
        # 检查极端值
        if torch.abs(tensor).max() > 1e10:
            self.log_warning(f"{name} 包含极端值，进行裁剪")
            tensor = torch.clamp(tensor, min=-1e10, max=1e10)
        
        return tensor
    
    def check_loss(self, loss: torch.Tensor) -> bool:
        """检查损失值是否合理"""
        if torch.isnan(loss) or torch.isinf(loss):
            self.stats['loss_exploded'] += 1
            self.log_warning("损失为 NaN 或 Inf")
            return False
        
        if loss.item() > self.max_loss_threshold:
            self.stats['loss_exploded'] += 1
            self.log_warning(f"损失值过大: {loss.item():.2e}")
            return False
        
        return True
    
    def check_gradients(self, model: torch.nn.Module) -> bool:
        """检查梯度是否合理"""
        total_norm = 0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                # 检查梯度中的NaN/Inf
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    self.stats['grad_exploded'] += 1
                    self.log_warning(f"参数梯度包含 NaN 或 Inf")
                    return False
                
                # 计算梯度范数
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            if total_norm > self.max_grad_norm:
                self.stats['grad_exploded'] += 1
                self.log_warning(f"梯度范数过大: {total_norm:.4f}")
                return False
        
        return True
    
    def apply_gradient_clipping(self, model: torch.nn.Module):
        """应用梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
    
    def check_model_parameters(self, model: torch.nn.Module) -> bool:
        """检查模型参数是否合理"""
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                self.stats['params_corrected'] += 1
                self.log_warning(f"模型参数 {name} 包含异常值，进行修正")
                
                # 重新初始化有问题的参数
                if param.dim() == 2:
                    torch.nn.init.xavier_uniform_(param.data, gain=0.01)
                else:
                    torch.nn.init.zeros_(param.data)
                
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'nan_detected': 0,
            'inf_detected': 0,
            'loss_exploded': 0,
            'grad_exploded': 0,
            'params_corrected': 0
        }


class LossStabilizer:
    """损失稳定器"""
    
    def __init__(self, eps: float = 1e-8, max_loss: float = 100.0):
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


class OptimizerStabilizer:
    """优化器稳定器"""
    
    def __init__(self, 
                 base_lr: float = 1e-4,
                 warmup_epochs: int = 3,
                 min_lr: float = 1e-7,
                 weight_decay: float = 1e-6):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.weight_decay = weight_decay
    
    def create_stable_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """创建稳定的优化器"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
    
    def get_lr_scale(self, epoch: int) -> float:
        """获取学习率缩放因子"""
        if epoch < self.warmup_epochs:
            # 学习率预热
            return (epoch + 1) / self.warmup_epochs
        else:
            # 学习率衰减
            return max(0.1, 0.99 ** (epoch - self.warmup_epochs))
    
    def update_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """更新学习率"""
        lr_scale = self.get_lr_scale(epoch)
        new_lr = max(self.base_lr * lr_scale, self.min_lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, patience: int = 10, min_improvement: float = 1e-6):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
    
    def update(self, loss: float) -> Dict[str, Any]:
        """更新监控状态"""
        self.training_history.append(loss)
        
        if loss < self.best_loss - self.min_improvement:
            self.best_loss = loss
            self.patience_counter = 0
            should_stop = False
            improved = True
        else:
            self.patience_counter += 1
            should_stop = self.patience_counter >= self.patience
            improved = False
        
        return {
            'should_stop': should_stop,
            'improved': improved,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.training_history:
            return {}
        
        return {
            'current_loss': self.training_history[-1],
            'best_loss': self.best_loss,
            'mean_loss': np.mean(self.training_history[-10:]),  # 最近10次的平均
            'loss_std': np.std(self.training_history[-10:]),
            'total_steps': len(self.training_history)
        }


def create_stable_training_environment():
    """创建稳定的训练环境"""
    # 设置PyTorch的数值稳定性
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 忽略警告
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    np.random.seed(42)
    
    return {
        'numerical_stabilizer': NumericalStabilizer(),
        'loss_stabilizer': LossStabilizer(),
        'optimizer_stabilizer': OptimizerStabilizer(),
        'training_monitor': TrainingMonitor()
    }
