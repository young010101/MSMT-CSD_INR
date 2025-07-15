import wandb
import torch
import nibabel as nib
import numpy as np
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque

from utils import parse_cfg
from loss_functions import get_loss_function
from output_calculators import get_output_calculator
from ml_utils import Trainer
from models import get_model
from datasets import get_dataset
from numerical_stability import NumericalStabilizer


class SmartTrainer:
    """智能训练器，包含损失监控和早期停止"""
    
    def __init__(self, cfg, device="cuda:5"):
        self.cfg = cfg
        self.device = device
        self.train_cfg = cfg["train_cfg"]
        
        # 早期停止参数
        self.patience = cfg.get("stability", {}).get("patience", 10)
        self.min_delta = 1e-6  # 最小改善阈值
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = deque(maxlen=20)  # 保存最近20个损失值
        
        # 损失趋势分析
        self.trend_window = 10  # 分析最近10个epoch的趋势
        self.improvement_threshold = 0.001  # 改善阈值
        
        # 初始化组件
        self._setup_components()
        
    def _setup_components(self):
        """设置训练组件"""
        # 创建模型
        self.model = get_model(self.cfg)
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建数据集和数据加载器
        self.dataset = get_dataset(self.cfg)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=3,
            drop_last=True,
        )
        
        # 创建损失函数和输出计算器
        self.loss_fn = get_loss_function(self.cfg)
        self.output_calculator = get_output_calculator(
            self.cfg, dataset=self.dataset, device=self.device
        )
        
        # 创建优化器和调度器
        weight_decay = self.cfg.get("stability", {}).get("weight_decay", 1e-6)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
            
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.train_cfg["lr"], 
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.8, 
            patience=5, 
            min_lr=1e-8
        )
        
        # 数值稳定器
        self.stabilizer = NumericalStabilizer()
        
        # 移动到设备
        self.model.to(self.device)
        
    def _analyze_loss_trend(self, current_loss):
        """分析损失趋势"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < self.trend_window:
            return {
                'slope': 0,
                'improvement': 0,
                'recent_avg': current_loss,
                'trend': 'insufficient_data'
            }
        
        # 计算最近几个epoch的趋势
        recent_losses = list(self.loss_history)[-self.trend_window:]
        
        # 计算线性回归斜率
        x = np.arange(len(recent_losses))
        y = np.array(recent_losses)
        
        # 简单的线性回归
        slope = np.polyfit(x, y, 1)[0]
        
        # 计算改善幅度
        if len(recent_losses) >= 2:
            improvement = recent_losses[0] - recent_losses[-1]
        else:
            improvement = 0
            
        return {
            'slope': slope,
            'improvement': improvement,
            'recent_avg': np.mean(recent_losses[-5:]),  # 最近5个epoch的平均值
            'trend': 'improving' if slope < -self.improvement_threshold else 'stagnant'
        }
    
    def _should_stop_early(self, current_loss, epoch):
        """判断是否应该早期停止"""
        # 基本早期停止检查
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            
        # 损失趋势分析
        trend_analysis = self._analyze_loss_trend(current_loss)
        
        # 如果损失趋势停滞且已经过了足够多的epoch
        if (trend_analysis.get('trend') == 'stagnant' and 
            epoch > 20 and 
            self.patience_counter >= self.patience // 2):
            print(f"损失趋势停滞，考虑早期停止")
            return True
            
        # 如果损失没有改善且已经过了很多epoch
        if (self.patience_counter >= self.patience and 
            epoch > 30):
            print(f"损失长期无改善，停止训练")
            return True
            
        # 如果损失开始上升
        if (len(self.loss_history) >= 5 and 
            current_loss > np.mean(list(self.loss_history)[-5:]) * 1.1):
            print(f"损失开始上升，停止训练")
            return True
            
        return False
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        losses = []
        nan_count = 0
        
        for i, (_input, labels) in enumerate(self.dataloader):
            _input = _input.to(self.device)
            labels = labels.to(self.device)
            
            # 检查输入数据
            if torch.isnan(_input).any() or torch.isinf(_input).any():
                nan_count += 1
                continue
                
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                nan_count += 1
                continue
            
            self.optimizer.zero_grad()
            
            try:
                # 前向传播
                model_out = self.model(_input)
                
                if torch.isnan(model_out).any() or torch.isinf(model_out).any():
                    nan_count += 1
                    continue
                
                # 计算输出和损失
                output, kwargs = self.output_calculator.output_from_model_out(model_out)
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    nan_count += 1
                    continue
                
                loss = self.loss_fn(output, labels, **kwargs)
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                    nan_count += 1
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.cfg.get("stability", {}).get("max_grad_norm", 1.0)
                )
                
                # 参数更新
                self.optimizer.step()
                
                losses.append(loss.item())
                
            except Exception as e:
                print(f"训练步骤出错: {e}")
                nan_count += 1
                continue
        
        if len(losses) == 0:
            return None, nan_count
            
        mean_loss = np.mean(losses)
        return mean_loss, nan_count
    
    def train(self):
        """主训练循环"""
        print("开始智能训练...")
        
        for epoch in range(self.train_cfg["epochs"]):
            # 训练一个epoch
            mean_loss, nan_count = self.train_epoch(epoch)
            
            if mean_loss is None:
                print(f"Epoch {epoch}: 没有有效批次，跳过")
                continue
            
            # 更新学习率调度器
            self.scheduler.step(mean_loss)
            
            # 分析损失趋势
            trend_analysis = self._analyze_loss_trend(mean_loss)
            
            # 打印训练信息
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: loss = {mean_loss:.6f}, "
                  f"lr = {current_lr:.2e}, NaN批次 = {nan_count}, "
                  f"趋势 = {trend_analysis.get('trend', 'N/A')}")
            
            # 检查是否应该早期停止
            if self._should_stop_early(mean_loss, epoch):
                print(f"早期停止在第 {epoch} 轮，最佳损失: {self.best_loss:.6f}")
                break
            
            # 每10个epoch保存一次检查点
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, mean_loss)
        
        # 保存最终模型
        self._save_final_model()
        
        return self.best_loss
    
    def _save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter
        }
        
        checkpoint_path = Path(self.cfg["paths"]["output"]) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
    
    def _save_final_model(self):
        """保存最终模型"""
        output_folder = Path(self.cfg["paths"]["output"])
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # 检查模型参数
        has_nan = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                print(f"警告：参数 {name} 包含 NaN 或 Inf")
                has_nan = True
                break
        
        if not has_nan:
            model_path = output_folder / "final_smart_model.pt"
            torch.save(self.model.state_dict(), model_path)
            print(f"最终模型已保存: {model_path}")
        else:
            print("警告：模型包含NaN值，跳过保存")


def main():
    """主函数"""
    # 设置数值稳定性
    torch.set_default_dtype(torch.float32)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 加载配置
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    
    # 创建智能训练器
    trainer = SmartTrainer(cfg)
    
    # 开始训练
    try:
        best_loss = trainer.train()
        print(f"训练完成，最佳损失: {best_loss:.6f}")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 