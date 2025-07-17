import wandb
import torch
import nibabel as nib
import time
import numpy as np
from torch.utils.data import Subset
import random
import argparse
import os
from pathlib import Path

from utils import parse_cfg

from loss_functions import get_loss_function
from output_calculators import get_output_calculator
from ml_utils import Trainer
from models import get_model
from datasets import get_dataset, DiffusionDataset
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# 环境变量控制是否启用wandb
USE_WANDB = os.environ.get("USE_WANDB", "0") == "1"

def split_dataset(dataset, strategy="odd_even", train_ratio=0.7, seed=42):
    """
    Split dataset according to different strategies
    
    Args:
        dataset: The dataset to split
        strategy: One of ["odd_even", "random"]
        train_ratio: Ratio of training data when using random strategy
        seed: Random seed for reproducibility
    
    Returns:
        train_indices, val_indices, test_indices
    """
    if isinstance(dataset, DiffusionDataset):
        input_tensor = dataset.input_tensor
    else:
        raise AttributeError('当前数据集类型不支持基于slice的划分（缺少input_tensor属性）')
    
    if strategy == "odd_even":
        # 基于z轴slice的奇偶性划分数据集
        depth = input_tensor.shape[2] if len(input_tensor.shape) > 2 else None
        if depth is None:
            # 如果input_tensor已经被reshape成2D，则无法确定depth
            # 尝试从z坐标推断
            z_coords = input_tensor[:, 2]
            z_idx = torch.round((z_coords + 1) * 72).to(torch.int).cpu().numpy()  # 假设depth=145
            train_indices = [i for i, z in enumerate(z_idx) if z % 2 == 1]
            val_indices = test_indices = [i for i, z in enumerate(z_idx) if z % 2 == 0]
        else:
            # 反归一化z坐标到索引（假设[-1,1] -> [0, depth-1]）
            z_coords = input_tensor[:, :, :, 2].reshape(-1)
            z_idx = torch.round((z_coords + 1) * (depth - 1) / 2).to(torch.int).cpu().numpy()
            train_indices = [i for i, z in enumerate(z_idx) if z % 2 == 1]
            val_indices = test_indices = [i for i, z in enumerate(z_idx) if z % 2 == 0]
        
    elif strategy == "random":
        # 随机划分
        random.seed(seed)
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_size = int(train_ratio * total_size)
        val_size = (total_size - train_size) // 2
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
        
    return train_indices, val_indices, test_indices

def initialize_run(config_path=None, cfg=None):
    if cfg is None:
        if config_path:
            cfg = parse_cfg(Path(config_path))
        else:
            cfg = parse_cfg(Path("configs/example_config.yaml"))
    
    # 启用wandb日志记录（如果需要）
    log_freq = cfg.get("log_freq", 0)
    if USE_WANDB and log_freq > 0:
        try:
            wandb.init(project=cfg["project_name"], job_type="training", config=cfg)
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            log_freq = 0  # 如果初始化失败，禁用日志记录
    else:
        log_freq = 0  # 禁用wandb
        print("Wandb logging disabled")
    
    train_cfg = cfg["train_cfg"]

    width = cfg["width"]
    height = cfg["height"]
    depth = cfg["depth"]

    l_max = train_cfg["lmax"]
    lr = train_cfg["lr"]
    lambda_ = train_cfg["lambda"]

    model = get_model(cfg)
    print(model)

    dataset = get_dataset(cfg)
    
    # 使用配置文件中指定的划分策略
    split_strategy = cfg.get("split_strategy", "odd_even")
    train_ratio = cfg.get("train_ratio", 0.7)
    train_indices, val_indices, test_indices = split_dataset(
        dataset, 
        strategy=split_strategy,
        train_ratio=train_ratio
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=3,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=3,
    )

    loss_fn = get_loss_function(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99) if cfg.get("scheduler", False) else None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    epochs = train_cfg["epochs"]

    output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        l_max=l_max,
        data_shape=(width, height, depth),
        output_calculator=output_calculator,
        log_freq=log_freq,
        lambda_=lambda_,
        slice_id=train_cfg["slice_id"],
        scheduler=scheduler,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    trainer.train()

    # 输出当前划分方式和数量统计
    print(f"\n==== 数据集划分方式：{split_strategy} ====")
    print(f"训练集体素数: {len(train_indices)}，验证集体素数: {len(val_indices)}，测试集体素数: {len(test_indices)}")
    
    # 输出volume序号、原始nifti索引和b值的对应关系表
    print("\n==== 输出volume序号与b值、原始nifti索引的对应关系 ====")
    if isinstance(dataset, DiffusionDataset):
        try:
            bvals = dataset.get_bvals()
            dwi_indices = dataset.get_dwi_idx()
            for i, (idx, bval) in enumerate(zip(dwi_indices, bvals)):
                print(f"输出volume序号: {i}, 原始nifti索引: {idx}, b值: {bval}")
        except Exception as e:
            print(f"无法输出volume与b值对应表: {e}")
    else:
        print("当前数据集类型不支持自动输出b值与volume索引表。")

    # 保存模型和结果
    file_inf = time.strftime("%Y%m%d_%H%M%S")
    if USE_WANDB and log_freq > 0 and wandb.run and hasattr(wandb.run, 'id'):
        file_inf = wandb.run.id  # 用wandb run id做唯一标识，若无则用时间戳

    output_folder = Path(cfg["paths"]["output"])
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    model_path = output_folder / f"model_{file_inf}.pt"
    torch.save(trainer.model.state_dict(), model_path)
    
    # 上传模型到wandb
    if USE_WANDB and log_freq > 0:
        try:
            wandb.save(str(model_path))
        except Exception as e:
            print(f"Warning: Failed to save model to wandb: {e}")

    try:
        if cfg["model_name"] in ["multishell", "split_multi"]:
            scale = float(dataset.get_scale()) if isinstance(dataset, DiffusionDataset) else 1.0
            (
                nifti_img,
                grad_img,
                coeff_image,
                gm_coeff,
                csf_coeff,
            ) = trainer.create_full_output_image(cfg, scale)

            gm_coeff_img = nib.Nifti1Image(
                gm_coeff, affine=nifti_img.affine, header=nifti_img.header
            )
            gm_coeff_path = output_folder / f"gm_coeffs_{file_inf}.nii.gz"
            nib.save(gm_coeff_img, gm_coeff_path)
            if USE_WANDB and log_freq > 0:
                try:
                    wandb.save(str(gm_coeff_path))
                except Exception as e:
                    print(f"Warning: Failed to save gm_coeffs to wandb: {e}")

            csf_coeff_img = nib.Nifti1Image(
                csf_coeff, affine=nifti_img.affine, header=nifti_img.header
            )
            csf_coeff_path = output_folder / f"csf_coeffs_{file_inf}.nii.gz"
            nib.save(csf_coeff_img, csf_coeff_path)
            if USE_WANDB and log_freq > 0:
                try:
                    wandb.save(str(csf_coeff_path))
                except Exception as e:
                    print(f"Warning: Failed to save csf_coeffs to wandb: {e}")
        else:
            scale = float(dataset.get_scale()) if isinstance(dataset, DiffusionDataset) else 1.0
            nifti_img, grad_img, coeff_image = trainer.create_full_output_image(
                cfg, scale
            )

        full_nifti_img = nib.Nifti1Image(
            grad_img, affine=nifti_img.affine, header=nifti_img.header
        )
        grads_path = output_folder / f"grads_{file_inf}.nii.gz"
        nib.save(full_nifti_img, grads_path)
        if USE_WANDB and log_freq > 0:
            try:
                wandb.save(str(grads_path))
            except Exception as e:
                print(f"Warning: Failed to save grads to wandb: {e}")

        full_coeff_img = nib.Nifti1Image(
            coeff_image, affine=nifti_img.affine, header=nifti_img.header
        )
        coeffs_path = output_folder / f"coeffs_{file_inf}.nii.gz"
        nib.save(full_coeff_img, coeffs_path)
        if USE_WANDB and log_freq > 0:
            try:
                wandb.save(str(coeffs_path))
            except Exception as e:
                print(f"Warning: Failed to save coeffs to wandb: {e}")
                
        # 为了方便测试，也保存一个固定名称的文件
        test_coeffs_path = output_folder / f"coeffs_test.nii.gz"
        nib.save(full_coeff_img, test_coeffs_path)
    except Exception as e:
        print(f"Warning: Failed to save output images: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with specified config")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--strategy", type=str, help="Training strategy to use (e.g. random_30, random_40, etc.)")
    args = parser.parse_args()
    
    if args.strategy:
        cfg = parse_cfg(Path(args.config))
        if args.strategy not in cfg:
            raise ValueError(f"Strategy {args.strategy} not found in config file")
        cfg = cfg[args.strategy]
        initialize_run(cfg=cfg)
    else:
        initialize_run(args.config)
