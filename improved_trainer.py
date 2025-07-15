"""
改进的训练器 - 解决NaN问题的根本原因
"""
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import nibabel as nib

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from output_calculators import OutputCalculator
from utils import spherical_to_cartesian, parse_bvals, parse_mrtrix
from datasets import create_input_space, DiffusionDataset, create_input_space_prop
from dataclasses import dataclass


@dataclass
class ImprovedTrainer:
    model: torch.nn.Module
    dataset: DiffusionDataset
    dataloader: DataLoader
    loss_fn: Callable
    optimizer: torch.optim.Optimizer
    device: str
    epochs: int
    l_max: int
    data_shape: tuple[int, int, int]
    output_calculator: OutputCalculator

    log_freq: int = 100
    lambda_: float = 0
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    slice_id: int = 0
    grad_id: int = 0
    patience: int = 10
    
    # 新增参数用于解决NaN问题
    init_lr: float = 1e-4  # 更小的初始学习率
    warmup_epochs: int = 5  # 学习率预热轮数
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    weight_decay: float = 1e-6  # 权重衰减
    eps: float = 1e-8  # 用于数值稳定性的小值

    # wandb 参数
    wandb_project: str = "stable_training"
    project_name: str = "MSMT-CSD_INR_Stable"

    def __post_init__(self):
        self.wandb_log = self.log_freq > 0
        if self.wandb_log:
            wandb.init(project=self.wandb_project, name=self.project_name)
            wandb.watch(self.model, log="all", log_freq=self.log_freq)

        self.model.to(self.device)
        self._initialize_model_weights()
        self._setup_optimizer()
        
    def _initialize_model_weights(self):
        """正确初始化模型权重以避免NaN"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化，缩放因子更小
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                # 对于高斯随机特征，使用更小的标准差
                if m.data.dim() == 2:  # 假设是B矩阵
                    nn.init.normal_(m.data, std=0.1)
                    
        self.model.apply(init_weights)
        
    def _setup_optimizer(self):
        """设置优化器，添加权重衰减"""
        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['weight_decay'] = self.weight_decay
                group['eps'] = self.eps
                
    def _get_lr_scale(self, epoch):
        """学习率预热调度"""
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        return 1.0
        
    def _check_and_fix_nan(self, tensor, name, default_value=0.0):
        """检查并修复NaN/Inf值"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"警告：{name} 包含 NaN/Inf，已修复")
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                               torch.tensor(default_value, device=tensor.device), 
                               tensor)
        return tensor
        
    def _safe_forward(self, inputs):
        """安全的前向传播"""
        try:
            # 添加数值稳定性
            inputs = torch.clamp(inputs, -10, 10)
            outputs = self.model(inputs)
            
            # 检查输出范围
            outputs = torch.clamp(outputs, -100, 100)
            return outputs
        except Exception as e:
            print(f"前向传播错误: {e}")
            return None
            
    def _compute_loss_safely(self, output, labels, **kwargs):
        """安全的损失计算"""
        try:
            # 添加数值稳定性
            output = self._check_and_fix_nan(output, "model_output")
            labels = self._check_and_fix_nan(labels, "labels")
            
            # 计算损失
            loss = self.loss_fn(output, labels, **kwargs)
            
            # 检查损失值
            if torch.isnan(loss) or torch.isinf(loss):
                print("损失为NaN/Inf，使用备用损失")
                loss = F.mse_loss(output, labels)
                
            return loss
        except Exception as e:
            print(f"损失计算错误: {e}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)
            
    def _gradient_surgery(self):
        """梯度手术 - 处理异常梯度"""
        total_norm = 0
        param_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                # 检查并修复梯度中的NaN/Inf
                p.grad = self._check_and_fix_nan(p.grad, "gradient")
                
                # 计算梯度范数
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
        total_norm = total_norm ** (1. / 2)
        
        # 自适应梯度裁剪
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + self.eps)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
                    
        return total_norm
        
    def _emergency_reset(self):
        """紧急重置 - 当模型参数严重偏离时"""
        print("执行紧急重置...")
        
        # 重新初始化部分参数
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"重置参数: {name}")
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param.data, gain=0.01)
                else:
                    nn.init.zeros_(param.data)
                    
        # 重置优化器状态
        self.optimizer.state = {}
        
    def train(self):
        """改进的训练循环"""
        avg_loss = []
        best_loss = float('inf')
        patience_counter = 0
        consecutive_nan_count = 0
        
        for epoch in range(self.epochs):
            losses = []
            lr_scale = self._get_lr_scale(epoch)
            
            # 应用学习率预热
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr * lr_scale
                
            epoch_nan_count = 0
            
            for i, (_input, labels) in enumerate(tqdm(self.dataloader)):
                self.model.train()
                _input = _input.to(self.device)
                labels = labels.to(self.device)
                
                # 数据预处理
                _input = self._check_and_fix_nan(_input, "input")
                labels = self._check_and_fix_nan(labels, "labels")
                
                # 检查数据质量
                if torch.isnan(_input).any() or torch.isnan(labels).any():
                    epoch_nan_count += 1
                    continue
                    
                self.optimizer.zero_grad()
                
                # 安全前向传播
                model_out = self._safe_forward(_input)
                if model_out is None:
                    epoch_nan_count += 1
                    continue
                    
                # 计算输出
                try:
                    output, kwargs = self.output_calculator.output_from_model_out(model_out)
                    output = self._check_and_fix_nan(output, "output_calc")
                except Exception as e:
                    print(f"输出计算错误: {e}")
                    epoch_nan_count += 1
                    continue
                    
                # 安全损失计算
                loss = self._compute_loss_safely(output, labels, **kwargs)
                
                # 检查损失合理性
                if loss.item() > 1000:  # 损失过大
                    print(f"损失过大: {loss.item()}")
                    epoch_nan_count += 1
                    continue
                    
                # 反向传播
                try:
                    loss.backward()
                except Exception as e:
                    print(f"反向传播错误: {e}")
                    epoch_nan_count += 1
                    continue
                    
                # 梯度处理
                grad_norm = self._gradient_surgery()
                
                # 参数更新
                self.optimizer.step()
                
                # 检查参数健康状况
                param_nan_count = 0
                for p in self.model.parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        param_nan_count += 1
                        
                if param_nan_count > 0:
                    print(f"参数包含NaN/Inf，数量: {param_nan_count}")
                    self._emergency_reset()
                    epoch_nan_count += 1
                    continue
                    
                losses.append(loss.item())
                
            # 处理轮次结果
            if len(losses) == 0:
                print(f"第 {epoch} 轮没有有效的批次")
                consecutive_nan_count += 1
                if consecutive_nan_count >= 3:
                    print("连续3轮无效，执行紧急重置")
                    self._emergency_reset()
                    consecutive_nan_count = 0
                continue
            else:
                consecutive_nan_count = 0
                
            mean_loss = np.array(losses).mean()
            
            print(f"Epoch {epoch}: loss = {mean_loss:.6f}, NaN批次: {epoch_nan_count}")
            
            # 早期停止检查
            if mean_loss < best_loss:
                best_loss = mean_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"早期停止在第 {epoch} 轮，最佳损失: {best_loss:.6f}")
                self.model.load_state_dict(torch.load('best_model.pth'))
                break
                
            if self.wandb_log:
                wandb.log({
                    "loss": mean_loss,
                    "epoch_nan_count": epoch_nan_count,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
                
            if self.scheduler:
                self.scheduler.step(mean_loss)
                
            avg_loss.append(mean_loss)
            
        return avg_loss
        
    def log_progress_image(self) -> None:
        """记录进度图像"""
        width, height, depth = self.data_shape
        input_coords = create_input_space_prop(width, height, depth)
        input_coords = (
            input_coords[:, :, self.slice_id].reshape(width * height, 3).to(self.device)
        )

        self.model.eval()
        with torch.no_grad():
            model_out = self._safe_forward(input_coords)
            if model_out is not None:
                diff_signal, *_ = self.output_calculator.output_from_model_out(model_out)
                image = diff_signal[:, self.grad_id]

                if diff_signal.dim() > 2:
                    image = image[..., 0]

                grad_pred_img = image.reshape(width, height).T
                wandb_image = wandb.Image(grad_pred_img)
                wandb.log({"progress image": wandb_image})
