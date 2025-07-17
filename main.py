import wandb
import torch
import nibabel as nib
import time
import numpy as np
from torch.utils.data import Subset

from utils import parse_cfg

from loss_functions import get_loss_function
from output_calculators import get_output_calculator
from ml_utils import Trainer
from models import get_model
from datasets import get_dataset, DiffusionDataset
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def initialize_run():
    cfg = parse_cfg(Path("configs/example_config.yaml"))
    # 启用wandb日志记录
    wandb.init(project=cfg["project_name"], job_type="training", config=cfg)
    # cfg = wandb.config  # 可选：用wandb.config替换cfg
    train_cfg = cfg["train_cfg"]

    width = cfg["width"]
    height = cfg["height"]
    depth = cfg["depth"]

    l_max = train_cfg["lmax"]
    lr = train_cfg["lr"]
    lambda_ = train_cfg["lambda"]
    log_freq = cfg["log_freq"]

    model = get_model(cfg)
    print(model)

    dataset: DiffusionDataset | Dataset = get_dataset(cfg)
    # 基于z轴slice的奇偶性划分数据集
    if isinstance(dataset, DiffusionDataset):
        input_tensor = dataset.input_tensor
    else:
        raise AttributeError('当前数据集类型不支持基于slice的划分（缺少input_tensor属性）')
    depth = cfg["depth"]
    # 反归一化z坐标到索引（假设[-1,1] -> [0, depth-1]）
    z_coords = input_tensor[:, 2]
    z_idx = torch.round((z_coords + 1) * (depth - 1) / 2).to(torch.int).cpu().numpy()
    train_indices = [i for i, z in enumerate(z_idx) if z % 2 == 1]
    test_indices = [i for i, z in enumerate(z_idx) if z % 2 == 0]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, test_indices)  # 偶数层既做val也做test
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
    print(f"\n==== 数据集划分方式：z轴奇数层用于训练，偶数层用于验证/测试 ====")
    print(f"训练集体素数: {len(train_indices)}，验证/测试集体素数: {len(test_indices)}")
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

    file_inf = wandb.run.id if wandb.run and hasattr(wandb.run, 'id') else time.strftime("%Y%m%d_%H%M%S")  # 用wandb run id做唯一标识，若无则用时间戳

    output_folder = Path(cfg["paths"]["output"])
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    model_path = output_folder / f"model_{file_inf}.pt"
    torch.save(trainer.model.state_dict(), model_path)
    # 上传模型到wandb
    wandb.save(str(model_path))

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
        wandb.save(str(gm_coeff_path))

        csf_coeff_img = nib.Nifti1Image(
            csf_coeff, affine=nifti_img.affine, header=nifti_img.header
        )
        csf_coeff_path = output_folder / f"csf_coeffs_{file_inf}.nii.gz"
        nib.save(csf_coeff_img, csf_coeff_path)
        wandb.save(str(csf_coeff_path))
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
    wandb.save(str(grads_path))

    full_coeff_img = nib.Nifti1Image(
        coeff_image, affine=nifti_img.affine, header=nifti_img.header
    )
    coeffs_path = output_folder / f"coeffs_{file_inf}.nii.gz"
    nib.save(full_coeff_img, coeffs_path)
    wandb.save(str(coeffs_path))


if __name__ == "__main__":
    initialize_run()
