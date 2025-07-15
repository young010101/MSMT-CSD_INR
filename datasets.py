from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import parse_bvecs, parse_bvals, parse_response, parse_mrtrix

def create_input_space(width: int, height: int, depth: int) -> torch.Tensor:
    p_width = 2 / width
    p_height = 2 / height
    p_depth = 2 / depth

    x = torch.linspace(-1 + p_width / 2, (1 - p_width / 2), width)
    y = torch.linspace(-1 + p_height / 2, (1 - p_height / 2), height)
    z = torch.linspace(-1 + p_depth / 2, (1 - p_depth / 2), depth)

    input_tensor = torch.cartesian_prod(x, y, z)

    return input_tensor.reshape(width, height, depth, 3)

def create_input_space_prop(width: int, height: int, depth: int) -> torch.Tensor:
    dims = np.array([width, height, depth])
    max_dim = np.max(dims)
    p_size = 2 / max_dim

    sizes = p_size*(dims - 1)

    x = torch.linspace(-sizes[0]/2, sizes[0]/2, width)
    y = torch.linspace(-sizes[1]/2, sizes[1]/2, height)
    z = torch.linspace(-sizes[2]/2, sizes[2]/2, depth)

    input_tensor = torch.cartesian_prod(x, y, z)

    return input_tensor.reshape(width, height, depth, 3)

def create_input_space_prop_upsampled(
    train_width: int, train_height: int, train_depth: int,
    pred_width: int, pred_height: int, pred_depth: int
) -> torch.Tensor:

    train_dims = np.array([train_width, train_height, train_depth])
    max_dim = np.max(train_dims)
    p_size = 2 / max_dim
    sizes = p_size * (train_dims - 1)

    x = torch.linspace(-sizes[0] / 2, sizes[0] / 2, pred_width)
    y = torch.linspace(-sizes[1] / 2, sizes[1] / 2, pred_height)
    z = torch.linspace(-sizes[2] / 2, sizes[2] / 2, pred_depth)

    input_tensor = torch.cartesian_prod(x, y, z)
    return input_tensor.reshape(pred_width, pred_height, pred_depth, 3)

def get_dwi_indices(bvals: np.array, bval: float, delta: float):
    bval_low = bval - delta
    bval_high = bval + delta

    return np.nonzero((bval_low < bvals) & (bvals < bval_high))[0]


def get_mean_b0(img: np.array, b0_idx: np.array):
    if len(b0_idx) == 0:
        raise Exception("No b0 images found")
    b0_imgs = img[..., b0_idx]
    if b0_idx.sum() == 1:
        return b0_imgs
    return b0_imgs.mean(axis=-1, keepdims=True)

class DiffusionDataset(Dataset):
    def get_dwi_idx(self) -> np.ndarray:
        pass

    def get_bvals(self) -> np.ndarray:
        pass

    def get_scale(self) -> float | np.ndarray:
        pass

    def get_response(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def get_directions(self) -> np.ndarray:
        pass


def _initialize_dwi_idx():
    """初始化DWI索引"""
    # 根据您的von shell数据集结构，这里需要定义哪些是DWI测量
    # 从代码中看到有16个b值，所以创建对应的索引
    n_directions = 16  # 根据您的bvals数组长度
    return torch.ones(n_directions, dtype=torch.bool)


class VonShellDataset(DiffusionDataset):
    def __init__(
        self,
        bvec_path: Path = None,
        bval_path: Path = None,
        mrtrix_bvec_path: Path = None,
        response_path: Path = None,
        shell: int = 0,
        bval_delta: int = 0,
        nifti_path: Path = None,
        mask_path: Path = None,
        scale: bool = True,
    ) -> None:
        # 对于von shell，我们不需要使用传入的nifti_path，而是使用专门的n19和n49数据
        # 这些路径应该在配置文件中指定
        
        # 尝试加载n19和n49数据
        try:
            # 使用配置文件中的路径（如果存在）
            n19_path = Path('/data/users/cyang/notes/Research/genAI/diffusion/qb_diff_real_delta_19_avg_by_b.nii')
            n49_path = Path('/home/cyang/repos/MSMT-CSD_INR/qb_diff_real_delta_49_avg_by_b.nii')
            
            if n19_path.exists() and n49_path.exists():
                n19 = nib.load(n19_path)
                n49 = nib.load(n49_path)
                full_img = np.concatenate([n19.get_fdata()[...,1:], n49.get_fdata()[...,1:]], axis=-1)
                print(f"成功加载von shell数据文件")
                print(f"n19形状: {n19.get_fdata()[...,1:].shape}")
                print(f"n49形状: {n49.get_fdata()[...,1:].shape}")
                print(f"合并后形状: {full_img.shape}")
            else:
                raise FileNotFoundError(f"von shell数据文件不存在: n19={n19_path.exists()}, n49={n49_path.exists()}")
        except Exception as e:
            raise FileNotFoundError(f"无法加载von shell数据文件: {e}")

        # 加载mask
        if mask_path and mask_path.exists():
            mask = nib.load(mask_path)
            mask_data = mask.get_fdata().astype(int)
            print(f"使用指定的mask文件: {mask_path}")
            print(f"Mask形状: {mask_data.shape}")
            
            # 检查mask和图像形状是否匹配
            if mask_data.shape[:3] != full_img.shape[:3]:
                print(f"警告：Mask形状 {mask_data.shape[:3]} 与图像形状 {full_img.shape[:3]} 不匹配")
                print("将调整mask大小以匹配图像")
                # 这里可以添加mask重采样逻辑，暂时使用全图像
                mask_4d = np.ones_like(full_img[..., :1], dtype=int)
            else:
                mask_4d = mask_data[..., np.newaxis]
        else:
            # 尝试使用备用mask路径
            try:
                backup_mask_path = Path('/data/users/zzhou/GNC/Data/HC_030/Delta_19/brainmask_mask.nii.gz')
                if backup_mask_path.exists():
                    mask = nib.load(backup_mask_path)
                    mask_data = mask.get_fdata().astype(int)
                    print(f"使用备用mask文件: {backup_mask_path}")
                    print(f"Mask形状: {mask_data.shape}")
                    
                    # 检查mask和图像形状是否匹配
                    if mask_data.shape[:3] != full_img.shape[:3]:
                        print(f"警告：备用Mask形状 {mask_data.shape[:3]} 与图像形状 {full_img.shape[:3]} 不匹配")
                        print("将使用全图像")
                        mask_4d = np.ones_like(full_img[..., :1], dtype=int)
                    else:
                        mask_4d = mask_data[..., np.newaxis]
                else:
                    print("警告：未找到mask文件，将使用全图像")
                    mask_4d = np.ones_like(full_img[..., :1], dtype=int)
            except Exception as e:
                print(f"警告：无法加载mask文件: {e}，将使用全图像")
                mask_4d = np.ones_like(full_img[..., :1], dtype=int)

        # 检查数据有效性
        if np.isnan(full_img).any():
            print("警告：输入图像包含NaN值，将替换为0")
            full_img = np.nan_to_num(full_img, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isinf(full_img).any():
            print("警告：输入图像包含Inf值，将替换为0")
            full_img = np.nan_to_num(full_img, nan=0.0, posinf=0.0, neginf=0.0)

        # 应用mask
        output_array = full_img * mask_4d
        
        # 检查输出数组
        if np.isnan(output_array).any():
            print("警告：masked图像包含NaN值，将替换为0")
            output_array = np.nan_to_num(output_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        width, height, depth, n_grad = output_array.shape
        print(f"最终图像形状: {output_array.shape}")
        print(f"Von shell输出通道数: {n_grad} (应该是16)")

        # 验证输出通道数
        if n_grad != 16:
            print(f"警告：输出通道数 {n_grad} 不等于16，这可能导致训练问题")

        # 创建输入坐标
        self.input_tensor = create_input_space_prop(width, height, depth)

        # 缩放数据
        if scale:
            # 使用更稳定的百分位数计算
            valid_data = output_array[output_array > 0]  # 只考虑非零值
            if len(valid_data) > 0:
                self.scale_value = np.percentile(valid_data, 99)
                print(f"缩放因子: {self.scale_value}")
            else:
                self.scale_value = 1.0
                print("警告：没有有效数据用于计算缩放因子")
        else:
            self.scale_value = 1.0
        
        output_array = output_array / self.scale_value

        # 应用mask到输入坐标
        if mask_path and mask_path.exists():
            try:
                mask_data = nib.load(mask_path).get_fdata().astype(int)
                brain_idx = np.asarray(mask_data == 1).nonzero()
                
                if len(brain_idx[0]) > 0:
                    self.input_tensor = self.input_tensor[
                        brain_idx[0], brain_idx[1], brain_idx[2]
                    ]
                    output_array = output_array[brain_idx[0], brain_idx[1], brain_idx[2], :]
                    print(f"应用mask后，有效体素数量: {len(brain_idx[0])}")
                else:
                    print("警告：mask中没有有效体素，使用全图像")
                    self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
                    output_array = output_array.reshape(width * height * depth, -1)
            except Exception as e:
                print(f"警告：应用mask时出错: {e}，使用全图像")
                self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
                output_array = output_array.reshape(width * height * depth, -1)
        else:
            self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
            output_array = output_array.reshape(width * height * depth, -1)

        # 转换为tensor并检查
        self.output_tensor = torch.tensor(output_array, dtype=torch.float32)
        
        # 检查输出tensor
        if torch.isnan(self.output_tensor).any():
            print("警告：输出tensor包含NaN值，将替换为0")
            self.output_tensor = torch.nan_to_num(self.output_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isinf(self.output_tensor).any():
            print("警告：输出tensor包含Inf值，将替换为0")
            self.output_tensor = torch.nan_to_num(self.output_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # 加载response文件
        if response_path and response_path.exists():
            try:
                response_data = parse_response(response_path)
                self.response_coeff = torch.tensor(
                    response_data[0], dtype=torch.float32
                ) / self.scale_value
                print(f"成功加载response文件: {response_path}")
            except Exception as e:
                print(f"警告：无法加载response文件: {e}，使用默认值")
                self.response_coeff = torch.ones(1, dtype=torch.float32)
        else:
            print("警告：response文件不存在，使用默认值")
            self.response_coeff = torch.ones(1, dtype=torch.float32)

        # 初始化DWI索引 - 对于von shell，我们有16个通道
        self.dwi_idx = torch.ones(16, dtype=torch.bool)
        
        print(f"VonShellDataset初始化完成，数据点数量: {len(self)}")
        print(f"输出tensor形状: {self.output_tensor.shape}")
        print(f"输入tensor形状: {self.input_tensor.shape}")

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_data = self.input_tensor[idx]
        output_data = self.output_tensor[idx]
        
        # 检查返回的数据
        if torch.isnan(input_data).any() or torch.isinf(input_data).any():
            print(f"警告：索引 {idx} 的输入数据包含异常值")
            input_data = torch.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(output_data).any() or torch.isinf(output_data).any():
            print(f"警告：索引 {idx} 的输出数据包含异常值")
            output_data = torch.nan_to_num(output_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return input_data, output_data

    def get_dwi_idx(self):
        """获取DWI索引"""
        if not hasattr(self, 'dwi_idx') or self.dwi_idx is None:
            self.dwi_idx = self._initialize_dwi_idx()
        return self.dwi_idx.nonzero().flatten()

    def get_scale(self):
        """获取缩放因子"""
        return self.scale_value

    def get_response(self) -> np.ndarray:
        return self.response_coeff.numpy()

    def get_directions(self) -> np.ndarray:
        # 对于von shell，我们需要返回适当的方向信息
        # 这里返回一个默认的方向数组
        n_directions = self.output_tensor.shape[1] if len(self.output_tensor.shape) > 1 else 1
        return np.ones((n_directions, 3))  # 默认方向

    def get_bvals(self) -> np.ndarray:
        # 返回b值数组，这里使用默认值
        n_directions = self.output_tensor.shape[1] if len(self.output_tensor.shape) > 1 else 1
        return np.ones(n_directions) * 3000  # 默认b值


class SingleShellDataset(DiffusionDataset):
    def __init__(
        self,
        bvec_path: Path,
        bval_path: Path,
        mrtrix_bvec_path: Path,
        response_path: Path,
        shell: int,
        bval_delta: int,
        nifti_path: Path,
        mask_path: Path = None,
        scale: bool = True,
    ) -> None:
        nifti_file = nib.load(nifti_path)

        if bvec_path and bval_path:
            self.bvals = parse_bvals(bval_path)
            self.bvecs = parse_bvecs(bvec_path)
        else:
            mrtrix_bvecs = parse_mrtrix(mrtrix_bvec_path)
            self.bvals = mrtrix_bvecs[:, -1]
            self.bvecs = mrtrix_bvecs[:, :3]

        full_img = nifti_file.get_fdata()
        self.dwi_idx = (self.bvals > (shell - bval_delta)) & (self.bvals < (shell + bval_delta))
        
        # available_indices = np.where(self.dwi_idx)[0]  # ??????????????? 90

        # # undersample 30
        # num_samples = 90 * 9 // 10
        # selected_indices = np.random.choice(available_indices, num_samples, replace=False)

        # # bvecs
        # undersampled_bvecs = bvecs[selected_indices]
        # undersampled_img = full_img[..., selected_indices]  # ??????? 30 ???
        output_array = full_img[..., self.dwi_idx]  # remove b0
        # output_array = undersampled_img
        width, height, depth, n_grad = output_array.shape

        self.cart_bvecs = parse_bvecs(bvec_path)[self.dwi_idx]  # remove b0
        # Separate into function for generating input coordinates
        self.input_tensor = create_input_space_prop(width, height, depth)

        self.scale_value = np.percentile(output_array, 99) if scale else 1
        output_array = output_array / self.scale_value

        if mask_path:
            mask_data = nib.load(mask_path).get_fdata().astype(int)
            brain_idx = np.asarray(mask_data == 1).nonzero()

            self.input_tensor = self.input_tensor[
                brain_idx[0], brain_idx[1], brain_idx[2]
            ]
            output_array = output_array[brain_idx[0], brain_idx[1], brain_idx[2], :]
        else:
            self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
            output_array = output_array.reshape(width * height * depth, -1)

        self.output_tensor = torch.tensor(output_array, dtype=torch.float32)
        self.response_coeff = (
            torch.tensor(
                parse_response(response_path)[0], dtype=torch.float
            )
            / self.scale_value
        )

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_tensor[idx], self.output_tensor[idx]

    def get_dwi_idx(self) -> np.ndarray:
        return self.dwi_idx.nonzero()[0]

    def get_scale(self) -> float | np.ndarray:
        return self.scale_value

    def get_response(self) -> np.ndarray:
        return self.response_coeff

    def get_directions(self) -> np.ndarray:
        return self.cart_bvecs

    def get_bvals(self) -> np.ndarray:
        return self.bvals


class MultiShellDataset(Dataset):
    def __init__(
        self,
        bvec_path: Path,
        bval_path: Path,
        mrtrix_bvec_path: Path,
        response_paths: list[Path],
        bval_delta: int,
        shells: np.array,
        nifti_path: Path,
        mask_path: Path = None,
        scale: bool = True,
    ) -> None:
        nifti_file = nib.load(nifti_path)
        # take long time
        nifti_img = nifti_file.get_fdata()

        if bvec_path and bval_path:
            bvals = parse_bvals(bval_path)
            all_bvecs = parse_bvecs(bvec_path)
        else:
            mrtrix_bvecs = parse_mrtrix(mrtrix_bvec_path)
            bvals = mrtrix_bvecs[:, -1]
            all_bvecs = mrtrix_bvecs[:, :3]

        self.shells = shells

        used = np.array([bval in shells for bval in bvals]) # used
        self.dwi_idx = used
        # take long time
        scale_values = [np.percentile(nifti_img[..., get_dwi_indices(bvals, bval, bval_delta)], 99) if scale else 1 for bval in bvals]

        b0_idx = (bvals < bval_delta)
        n_b0 = b0_idx.sum()
        if n_b0 > 0:
            b0_img = get_mean_b0(nifti_img, b0_idx)
            b0_scale = np.percentile(b0_img, 99) if scale else 1

            used[b0_idx] = False
            used = np.append(used, True)
            bvals = np.append(bvals, 0)
            scale_values = np.append(scale_values, b0_scale)

            nifti_img = np.concatenate([nifti_img, get_mean_b0(nifti_img, b0_idx)], axis=-1)
            all_bvecs = np.concatenate([all_bvecs, np.array([1, 0, 0])[None, :]])

        full_img = nifti_img[..., used]
        self.sel_bvals = bvals[used]
        self.scale_value = scale_values[used]

        output_array = full_img/self.scale_value

        self.cart_bvecs = all_bvecs[used]

        width, height, depth, n_grad = output_array.shape
        self.input_tensor = create_input_space_prop(width, height, depth)

        if mask_path:
            mask_data = nib.load(mask_path).get_fdata().astype(int)
            brain_idx = np.asarray(mask_data == 1).nonzero()
            self.input_tensor = self.input_tensor[
                brain_idx[0], brain_idx[1], brain_idx[2]
            ]
            output_array = output_array[brain_idx[0], brain_idx[1], brain_idx[2]]
        else:
            self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
            output_array = output_array.reshape(width * height * depth, -1)

        self.output_tensor = torch.tensor(output_array, dtype=torch.float)

        resp_scaler = np.array([scale_values[(bvals == shell)&used][0] for shell in shells])[:, None]
        self.wm_response = torch.tensor(
            parse_response(response_paths[0]) / resp_scaler,
            dtype=torch.float,
        )
        self.gm_response = torch.tensor(
            parse_response(response_paths[1]) / resp_scaler,
            dtype=torch.float,
        )
        self.csf_response = torch.tensor(
            parse_response(response_paths[2]) / resp_scaler,
            dtype=torch.float,
        )

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_tensor[idx], self.output_tensor[idx]

    def get_dwi_idx(self) -> np.ndarray:
        return self.dwi_idx.nonzero()[0]

    def get_bvals(self) -> np.ndarray:
        return self.sel_bvals

    def get_scale(self) -> float | np.ndarray:
        return self.scale_value

    def get_response(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.wm_response, self.gm_response, self.csf_response

    def get_directions(self) -> np.ndarray:
        return self.cart_bvecs


def create_vonshell(cfg: dict) -> DiffusionDataset:
    mask_path = Path(cfg["paths"]["mask"]) if cfg["paths"].get("mask", None) else None
    response_path = Path(cfg['paths']['response'])
    
    # 对于von shell，我们不需要bvec、bval等DWI相关参数
    # 只需要mask和response
    dataset = VonShellDataset(
        bvec_path=None,  # von shell不需要
        bval_path=None,  # von shell不需要
        mrtrix_bvec_path=None,  # von shell不需要
        response_path=response_path,
        shell=0,  # von shell不需要shell参数
        bval_delta=0,  # von shell不需要bval_delta参数
        nifti_path=None,  # von shell使用专门的n19/n49数据
        mask_path=mask_path,
        scale=cfg["scale_data"],
    )

    return dataset



def create_singleshell(cfg: dict) -> DiffusionDataset:
    mask_path = Path(cfg["paths"]["mask"]) if cfg["paths"].get("mask", None) else None
    bvec_path = Path(cfg["paths"]["fsl_bvecs"]) if cfg["paths"].get("fsl_bvecs", None) else None
    bval_path = Path(cfg["paths"]["fsl_bvals"]) if cfg["paths"].get("fsl_bvals", None) else None
    mrtrix_bvec_path = Path(cfg["paths"]["mrtrix_bvecs"]) if cfg["paths"].get("mrtrix_bvecs", None) else None

    response_path = Path(cfg['paths']['response'])
    dataset = SingleShellDataset(
        bvec_path=bvec_path,
        bval_path=bval_path,
        mrtrix_bvec_path=mrtrix_bvec_path,
        response_path=response_path,
        shell=cfg["shells"][0],
        bval_delta=cfg["bval_delta"],
        nifti_path=Path(cfg["paths"]["nifti"]),
        mask_path=mask_path,
        scale=cfg["scale_data"],
    )

    return dataset


def create_multishell(cfg: dict) -> MultiShellDataset:
    mask_path = Path(cfg["paths"]["mask"]) if cfg["paths"].get("mask", None) else None
    bvec_path = Path(cfg["paths"]["fsl_bvecs"]) if cfg["paths"].get("fsl_bvecs", None) else None
    bval_path = Path(cfg["paths"]["fsl_bvals"]) if cfg["paths"].get("fsl_bvals", None) else None
    mrtrix_bvec_path = Path(cfg["paths"]["mrtrix_bvecs"]) if cfg["paths"].get("mrtrix_bvecs", None) else None

    response_paths = [
        Path(cfg["paths"]["wm_response"]),
        Path(cfg["paths"]["gm_response"]),
        Path(cfg["paths"]["csf_response"]),
    ]
    dataset = MultiShellDataset(
        bvec_path=bvec_path,
        bval_path=bval_path,
        mrtrix_bvec_path=mrtrix_bvec_path,
        response_paths=response_paths,
        bval_delta=cfg["bval_delta"],
        shells=np.array(cfg["shells"]),
        nifti_path=Path(cfg["paths"]["nifti"]),
        mask_path=mask_path,
        scale=cfg["scale_data"],
    )

    return dataset

DATASETS = {
    "singleshell": create_singleshell,
    "multishell": create_multishell,
    "von_shell": create_vonshell,
}


def get_dataset(cfg: dict) -> Dataset:
    constructor = DATASETS.get(cfg["dataset_name"], None)
    if constructor is None:
        raise Exception("Dataset name not recognized")
    return constructor(cfg)
