import wandb
import torch
import nibabel as nib
import numpy as np
import warnings

from utils import parse_cfg
from loss_functions import get_loss_function
from output_calculators import get_output_calculator
from ml_utils import Trainer
from models import get_model
from datasets import get_dataset
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from numerical_stability import NumericalStabilizer


def check_nan_in_tensor(tensor, name="tensor"):
    """检查张量中是否包含NaN值"""
    if torch.isnan(tensor).any():
        print(f"警告：{name} 包含 NaN 值")
        return True
    if torch.isinf(tensor).any():
        print(f"警告：{name} 包含 Inf 值")
        return True
    return False


def safe_tensor_operation(tensor, operation_name="operation"):
    """安全的张量操作包装器"""
    try:
        if check_nan_in_tensor(tensor, f"{operation_name}_input"):
            return None
        return tensor
    except Exception as e:
        print(f"错误：{operation_name} 操作失败: {e}")
        return None


def initialize_run():
    # 设置数值稳定性
    torch.set_default_dtype(torch.float32)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    cfg = parse_cfg(Path("configs/example_config_von.yaml"))
    #wandb.init(project=cfg["project_name"], job_type="testing", config=cfg)

    #cfg = wandb.config
    train_cfg = cfg["train_cfg"]

    width = cfg["width"]
    height = cfg["height"]
    depth = cfg["depth"]

    l_max = train_cfg["lmax"]
    lr = train_cfg["lr"]
    lambda_ = train_cfg["lambda"]
    log_freq = cfg["log_freq"]

    # 创建数值稳定器
    stabilizer = NumericalStabilizer()

    # 创建模型并检查
    model = get_model(cfg)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 检查模型参数是否包含NaN
    for name, param in model.named_parameters():
        if check_nan_in_tensor(param.data, f"模型参数 {name}"):
            print(f"初始化时发现NaN参数: {name}")
            # 重新初始化有问题的参数
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                torch.nn.init.zeros_(param.data)

    dataset = get_dataset(cfg)
    print("数据集加载完成")
    
    # 检查数据集
    sample_input, sample_output = dataset[0]
    if check_nan_in_tensor(sample_input, "样本输入"):
        raise ValueError("数据集输入包含NaN值")
    if check_nan_in_tensor(sample_output, "样本输出"):
        raise ValueError("数据集输出包含NaN值")
    
    print(f"输入形状: {sample_input.shape}, 输出形状: {sample_output.shape}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )

    loss_fn = get_loss_function(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99) if cfg.get("scheduler", False) else None

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    epochs = train_cfg["epochs"]

    output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        dataloader=dataloader,
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
    )

    # 开始训练
    try:
        trainer.train()
        print("训练完成")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

    file_inf = "test"

    output_folder = Path(cfg["paths"]["output"])
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    
    # 保存模型前检查
    model_state = trainer.model.state_dict()
    has_nan = False
    for name, param in model_state.items():
        if check_nan_in_tensor(param, f"保存前模型参数 {name}"):
            has_nan = True
            break
    
    # 改进模型命名：包含配置信息
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = cfg["model_name"]
    lr_str = str(train_cfg["lr"]).replace(".", "p")
    batch_size = train_cfg["batch_size"]
    
    model_filename = f"model_{config_name}_lr{lr_str}_bs{batch_size}_{timestamp}.pt"
    
    if not has_nan:
        torch.save(model_state, output_folder / model_filename)
        print(f"模型已保存到: {output_folder / model_filename}")
    else:
        print("警告：模型包含NaN值，跳过保存")

    if cfg["model_name"] in ["multishell", "split_multi"]:
        (
            nifti_img,
            grad_img,
            coeff_image,
            gm_coeff,
            csf_coeff,
        ) = trainer.create_full_output_image(cfg, dataset.get_scale())

        gm_coeff_img = nib.Nifti1Image(
            gm_coeff, affine=nifti_img.affine, header=nifti_img.header
        )
        nib.save(gm_coeff_img, output_folder / f"gm_coeffs_{config_name}_{timestamp}.nii.gz")

        csf_coeff_img = nib.Nifti1Image(
            csf_coeff, affine=nifti_img.affine, header=nifti_img.header
        )
        nib.save(csf_coeff_img, output_folder / f"csf_coeffs_{config_name}_{timestamp}.nii.gz")
    else:
        nifti_img, grad_img, coeff_image = trainer.create_full_output_image(
            cfg, dataset.get_scale()
        )

    full_nifti_img = nib.Nifti1Image(
        grad_img, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(full_nifti_img, output_folder / f"grads_{config_name}_{timestamp}.nii.gz")

    full_coeff_img = nib.Nifti1Image(
        coeff_image, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(full_coeff_img, output_folder / f"coeffs_{config_name}_{timestamp}.nii.gz")


if __name__ == "__main__":
    initialize_run()
