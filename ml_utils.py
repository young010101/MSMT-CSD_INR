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
        for epoch in range(self.epochs):
            losses = []
            best_loss = float('inf')
            patience_counter = 0
            step = 0
            for i, (_input, labels) in enumerate(tqdm(self.dataloader)):
                self.model.train()
                _input = _input.to(self.device)

                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                model_out = self.model(
                    _input
                )  # [B, n_dir*3] Theta, Phi, volume fraction

                # 🔍 实时健康监控
                health_stats = self.health_monitor.analyze_model_output(model_out, step)

                # 根据健康评分决定是否打印详细报告
                if health_stats['health_score'] < 70:
                # if step % 10 == 0 or health_stats['health_score'] < 70:
                        self.health_monitor.print_health_report(health_stats)

                # 健康评分过低时的紧急措施
                if health_stats['health_score'] < 30:
                    print("🚨 健康评分过低，执行紧急干预！")
                    self._emergency_intervention()
                    continue

                # 检查模型输出
                if torch.isnan(model_out).any():
                    print(f"警告：模型输出包含 NaN，第 {epoch} 轮第 {i} 批次")
                    print(
                        f"模型输出统计: min={model_out.min():.6f}, max={model_out.max():.6f}, mean={model_out.mean():.6f}")

                    # 重新初始化模型参数
                    def reinit_weights(m):
                        if isinstance(m, torch.nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
                            torch.nn.init.zeros_(m.bias)

                    print("重新初始化模型参数...")
                    self.model.apply(reinit_weights)
                    continue

                # todo: map and (fh, fr, a, Dh)
                output, kwargs = self.output_calculator.output_from_model_out(model_out)
                loss = self.loss_fn(output, labels, **kwargs)

                loss.backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
                self.optimizer.step()

                loss_item = loss.item()
                losses.append(loss_item)
                step += 1

            mean_loss = np.array(losses).mean()

            # 早期停止检查
            if mean_loss < best_loss:
                best_loss = mean_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

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
