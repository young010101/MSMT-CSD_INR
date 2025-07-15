from typing import Any, Callable

import numpy as np
import torch

import wandb
import nibabel as nib

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from output_calculators import OutputCalculator
from utils import spherical_to_cartesian, parse_bvals, parse_mrtrix
from datasets import create_input_space, DiffusionDataset, create_input_space_prop
from dataclasses import dataclass
from health_monitor import health_monitor


@dataclass
class Trainer:
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
    patience: int = 10  # 早期停止的耐心值

    def __post_init__(self):
        self.wandb_log = self.log_freq > 0
        if self.wandb_log:
            wandb.watch(self.model, log="all", log_freq=self.log_freq)

        self.model.to(self.device)
        self.health_monitor = health_monitor

    def train(self):
        avg_loss = []
        best_loss = float('inf')
        patience_counter = 0
        nan_counter = 0
        max_nan_epochs = 5  # 最大允许连续出现nan的epoch数
        
        for epoch in range(self.epochs):
            losses = []
            step = 0
            epoch_nan_count = 0
            
            print(f"\n开始第 {epoch} 轮训练...")
            
            for i, (_input, labels) in enumerate(tqdm(self.dataloader)):
                self.model.train()
                _input = _input.to(self.device)
                labels = labels.to(self.device)
                
                # 检查输入数据
                if torch.isnan(_input).any() or torch.isinf(_input).any():
                    print(f"警告：输入数据包含 NaN 或 Inf，跳过此批次")
                    epoch_nan_count += 1
                    continue
                
                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    print(f"警告：标签数据包含 NaN 或 Inf，跳过此批次")
                    epoch_nan_count += 1
                    continue
                
                self.optimizer.zero_grad()
                
                try:
                    model_out = self.model(_input)
                    
                    # 检查模型输出
                    if torch.isnan(model_out).any() or torch.isinf(model_out).any():
                        print(f"警告：模型输出包含 NaN 或 Inf，跳过此批次")
                        epoch_nan_count += 1
                        continue
                    
                    # 计算输出和损失
                    output, kwargs = self.output_calculator.output_from_model_out(model_out)
                    
                    # 检查输出计算结果
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"警告：输出计算结果包含 NaN 或 Inf，跳过此批次")
                        epoch_nan_count += 1
                        continue
                    
                    loss = self.loss_fn(output, labels, **kwargs)
                    
                    # 检查损失
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告：损失为 NaN 或 Inf，跳过此批次")
                        epoch_nan_count += 1
                        continue
                    
                    # 检查损失值是否过大
                    if loss.item() > 100:  # 更严格的损失阈值
                        print(f"警告：损失值过大 ({loss.item():.2e})，跳过此批次")
                        epoch_nan_count += 1
                        continue
                    
                    loss.backward()
                    
                    # 检查梯度
                    total_norm = 0
                    has_nan_grad = False
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                print(f"警告：梯度包含 NaN 或 Inf，跳过此批次")
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        self.optimizer.zero_grad()
                        epoch_nan_count += 1
                        continue
                    
                    total_norm = total_norm ** (1. / 2)
                    
                    # 梯度裁剪
                    if total_norm > 1.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        print(f"应用梯度裁剪，梯度范数: {total_norm:.4f}")
                    
                    # 检查参数更新前是否包含NaN
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                            print(f"警告：参数 {name} 包含 NaN 或 Inf，跳过此批次")
                            epoch_nan_count += 1
                            break
                    else:
                        # 如果没有NaN，则更新参数
                        self.optimizer.step()
                        
                        loss_item = loss.item()
                        losses.append(loss_item)
                        step += 1
                        
                        # 每100步打印一次损失
                        if step % 100 == 0:
                            print(f"Step {step}, Loss: {loss_item:.6f}")
                
                except Exception as e:
                    print(f"训练步骤出错: {e}")
                    epoch_nan_count += 1
                    continue

            if len(losses) == 0:
                print(f"第 {epoch} 轮没有有效的批次，跳过此轮")
                nan_counter += 1
                if nan_counter >= max_nan_epochs:
                    print(f"连续 {max_nan_epochs} 轮出现NaN，停止训练")
                    break
                continue
                
            mean_loss = np.array(losses).mean()
            print(f"Epoch {epoch}: loss = {mean_loss:.6f}, 有效批次: {len(losses)}/{len(self.dataloader)}")
            
            # 重置nan计数器
            if epoch_nan_count == 0:
                nan_counter = 0
            else:
                print(f"本轮跳过 {epoch_nan_count} 个批次")

            # 早期停止检查
            if mean_loss < best_loss:
                best_loss = mean_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"保存最佳模型，损失: {best_loss:.6f}")
            else:
                patience_counter += 1
                print(f"损失未改善，耐心计数: {patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print(f"早期停止在第 {epoch} 轮，最佳损失: {best_loss:.6f}")
                # 加载最佳模型
                self.model.load_state_dict(torch.load('best_model.pth'))
                break

            if self.wandb_log:
                wandb.log({"loss": mean_loss})
                self.log_progress_image()

            if self.scheduler:
                self.scheduler.step()

            avg_loss.append(mean_loss)
            
            # 每5轮检查一次模型参数
            if epoch % 5 == 0:
                self._check_model_health()

    def log_progress_image(self) -> None:
        width, height, depth = self.data_shape
        input_coords = create_input_space_prop(width, height, depth)
        input_coords = (
            input_coords[:, :, self.slice_id].reshape(width * height, 3).to(self.device)
        )

        self.model.eval()
        with torch.no_grad():
            model_out = self.model(input_coords)
        diff_signal, *_ = self.output_calculator.output_from_model_out(model_out)
        image = diff_signal[:, self.grad_id]

        if diff_signal.dim() > 2:
            image = image[..., 0]

        grad_pred_img = image.reshape(width, height).T

        wandb_image = wandb.Image(grad_pred_img)
        wandb.log({"progress image": wandb_image})

    def create_full_output_image(self, cfg: dict, rescale_value: float = 1) -> Any:
        # 对于von shell配置，使用mask路径作为参考
        if cfg["model_name"] == "von_shell":
            # 使用mask文件作为参考
            mask_path = cfg["paths"]["mask"]
            nifti_img = nib.load(Path(mask_path))
            grad_img = nifti_img.get_fdata()
            width, height, depth = grad_img.shape[:3]
        else:
            # 原有的DWI配置
            nifti_img = nib.load(Path(cfg["paths"]["recon_nifti"]))
            grad_img = nifti_img.get_fdata()
            width, height, depth = grad_img.shape[:3]

        input_coords = create_input_space_prop(width, height, depth)
        input_coords = input_coords.reshape(width * height * depth, 3).to(self.device)

        coeff_outputs = []
        image_outputs = []

        self.model.eval()
        with torch.no_grad():
            for chunk in input_coords.split(1000):
                coeff_output = self.model(chunk)
                coeff_outputs.append(coeff_output.cpu().detach())
                image_out, *_ = self.output_calculator.output_from_model_out(
                    coeff_output
                )
                image_outputs.append(image_out.cpu().detach())

        diff_image = torch.concat(image_outputs)
        coeff_image = torch.concat(coeff_outputs)

        grad_pred_img = diff_image.reshape(
            width, height, depth, -1
        ).numpy()
        grad_pred_img_rescale = grad_pred_img * rescale_value
        coeff_image = coeff_image.reshape(width, height, depth, -1).numpy()

        # 对于von shell，使用mask
        if cfg["model_name"] == "von_shell":
            mask_img = nib.load(Path(cfg["paths"]["mask"])).get_fdata()
            grad_pred_img_rescale *= mask_img[:, :, :, None]
            coeff_image *= mask_img[:, :, :, None]
            return nifti_img, grad_pred_img_rescale, coeff_image
        else:
            # 原有的DWI处理逻辑
            mask_path = cfg["paths"].get("recon_mask", None)
            if mask_path:
                mask_img = nib.load(mask_path).get_fdata()
                grad_pred_img_rescale *= mask_img[:, :, :, None]
                coeff_image *= mask_img[:, :, :, None]

            if cfg["model_name"] in ["multishell", "split_multi"]:
                return (
                    nifti_img,
                    np.roll(grad_pred_img_rescale, 1, -1),
                    coeff_image[..., :-2],
                    coeff_image[..., -2],
                    coeff_image[..., -1],
                )

            if cfg["paths"].get("fsl_bvals", None):
                bvals = parse_bvals(Path(cfg["paths"]["fsl_bvals"]))
            else:
                bvals = parse_mrtrix(Path(cfg["paths"]["mrtrix_bvecs"]))[:, -1]

            b0_idx = (bvals < cfg["bval_delta"]).nonzero()[0]
            used_volumes = self.dataset.get_dwi_idx()

            for i, idx in enumerate(used_volumes):
                grad_img[..., idx] = grad_pred_img_rescale[
                    ..., i
                ]

            all_volumes = np.sort(np.concatenate([b0_idx, used_volumes]))
            grad_img = grad_img[..., all_volumes]

            return nifti_img, grad_img, coeff_image

    def _check_model_health(self):
        """检查模型健康状态"""
        print("检查模型健康状态...")
        total_params = 0
        nan_params = 0
        inf_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if torch.isnan(param.data).any():
                nan_params += param.numel()
                print(f"参数 {name} 包含 NaN")
            if torch.isinf(param.data).any():
                inf_params += param.numel()
                print(f"参数 {name} 包含 Inf")
        
        if nan_params > 0 or inf_params > 0:
            print(f"警告：模型包含 {nan_params} 个NaN参数和 {inf_params} 个Inf参数")
        else:
            print(f"模型健康，总参数数: {total_params}")
