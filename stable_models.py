"""
数值稳定的模型定义
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
from functools import partial
from numerical_stability import ActivationStabilizer, ModelInitializer


def stable_positional_encoding(
    x: torch.Tensor, lpos: int, sigma: float = None, max_freq: float = 10.0
) -> torch.Tensor:
    """数值稳定的位置编码"""
    if not sigma:
        js = 2 ** torch.arange(lpos, dtype=torch.float32).to(x.device) * torch.pi
    else:
        js = 2 ** torch.linspace(0, sigma, lpos, dtype=torch.float32).to(x.device) * torch.pi
    
    # 限制频率以避免数值不稳定
    js = torch.clamp(js, max=max_freq)
    
    jx = torch.einsum("ix, j -> ijx", x, js)
    sin_out = torch.sin(jx).reshape(x.shape[0], -1)
    cos_out = torch.cos(jx).reshape(x.shape[0], -1)
    
    return torch.cat([x, sin_out, cos_out], dim=-1)


def stable_input_mapping(x, B=None, max_freq: float = 10.0):
    """数值稳定的输入映射"""
    if B is None:
        return x
    else:
        # 限制B的值避免极端情况
        B_clamped = torch.clamp(B, -max_freq, max_freq)
        x_proj = (2.0 * torch.pi * x) @ B_clamped.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class StableLinear(nn.Module):
    """数值稳定的线性层"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x):
        # 裁剪输入防止极端值
        x = torch.clamp(x, -10, 10)
        output = self.linear(x)
        # 裁剪输出防止爆炸
        return torch.clamp(output, -100, 100)


class StableActivation(nn.Module):
    """数值稳定的激活函数"""
    
    def __init__(self, activation_type: str = "relu", max_val: float = 20.0):
        super().__init__()
        self.activation_type = activation_type
        self.max_val = max_val
        
    def forward(self, x):
        if self.activation_type == "relu":
            return ActivationStabilizer.stable_relu(x, self.max_val)
        elif self.activation_type == "tanh":
            return ActivationStabilizer.stable_tanh(x)
        elif self.activation_type == "sigmoid":
            return ActivationStabilizer.stable_sigmoid(x)
        else:
            return F.relu(torch.clamp(x, max=self.max_val))


class StableFod_NeSH(nn.Module):
    """数值稳定的Fod_NeSH模型"""
    
    def __init__(
        self,
        l_max=8,
        lpos=10,
        hidden_dim=256,
        n_layers=11,
        sigma=None,
        gaussian=True,
        dropout_rate=0.1,
        max_freq=5.0,  # 降低最大频率
    ) -> None:
        super().__init__()

        output_size = (l_max + 1) * (l_max + 2) // 2
        input_size = lpos * 2 if gaussian else lpos * 6 + 3
        
        # 构建更稳定的网络
        layers = []
        
        # 输入层
        layers.append(StableLinear(input_size, hidden_dim))
        layers.append(StableActivation("relu"))
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(n_layers - 1):
            layers.append(StableLinear(hidden_dim, hidden_dim))
            layers.append(StableActivation("relu"))
            if i % 2 == 0:  # 每隔一层添加dropout
                layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(StableLinear(hidden_dim, output_size))
        
        self.mlp = nn.Sequential(*layers)
        
        self.Lpos = lpos
        self.sigma = sigma
        self.max_freq = max_freq
        
        # 初始化高斯随机特征
        if gaussian:
            self.B = nn.Parameter(torch.randn([lpos, 3]) * 0.1)  # 更小的初始化
        else:
            self.B = None
            
        # 应用权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m, std=0.01)

    def forward(self, x, t_frac=None) -> torch.Tensor:
        # 输入预处理
        x = torch.clamp(x, -10, 10)
        
        # 特征编码
        if self.B is not None:
            # 确保B的值在合理范围内
            B_clamped = torch.clamp(self.B, -self.max_freq, self.max_freq)
            x_emb = stable_input_mapping(x, B_clamped, self.max_freq)
        else:
            x_emb = stable_positional_encoding(x, self.Lpos, self.sigma, self.max_freq)

        # 渐进式编码（如果需要）
        if t_frac is not None:
            x_emb = self._progressive_emb(x_emb, t_frac)

        # 前向传播
        output = self.mlp(x_emb)
        
        # 输出后处理
        return torch.clamp(output, -50, 50)
        
    def _progressive_emb(self, x_emb: torch.Tensor, t_frac: float) -> torch.Tensor:
        """渐进式编码"""
        a = torch.ones(x_emb.shape[1], device=x_emb.device)
        start = int(t_frac * x_emb.shape[1] + 3)
        end = int(t_frac * x_emb.shape[1] + 4)
        
        if start < x_emb.shape[1]:
            a[start:end] = (t_frac * x_emb.shape[1]) - int(t_frac * x_emb.shape[1])
        if end < x_emb.shape[1]:
            a[end:] = 0

        return x_emb * a.unsqueeze(dim=0)


class StableVon_G(nn.Module):
    """数值稳定的Von_G模型"""
    
    def __init__(
        self,
        l_max=8,
        lpos=10,
        hidden_dim=256,
        n_layers=11,
        sigma=None,
        gaussian=True,
        dropout_rate=0.1,
        max_freq=5.0,
    ) -> None:
        super().__init__()

        output_size = 4
        input_size = lpos * 2 if gaussian else lpos * 6 + 3
        
        # 构建稳定的网络
        layers = []
        
        # 输入层
        layers.append(StableLinear(input_size, hidden_dim))
        layers.append(StableActivation("relu"))
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(n_layers - 1):
            layers.append(StableLinear(hidden_dim, hidden_dim))
            layers.append(StableActivation("relu"))
            if i % 2 == 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(StableLinear(hidden_dim, output_size))
        
        self.mlp = nn.Sequential(*layers)
        
        self.Lpos = lpos
        self.sigma = sigma
        self.max_freq = max_freq
        
        if gaussian:
            self.B = nn.Parameter(torch.randn([lpos, 3]) * 0.1)
        else:
            self.B = None
            
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, t_frac=None) -> torch.Tensor:
        x = torch.clamp(x, -10, 10)
        
        if self.B is not None:
            B_clamped = torch.clamp(self.B, -self.max_freq, self.max_freq)
            x_emb = stable_input_mapping(x, B_clamped, self.max_freq)
        else:
            x_emb = stable_positional_encoding(x, self.Lpos, self.sigma, self.max_freq)

        if t_frac is not None:
            x_emb = self._progressive_emb(x_emb, t_frac)

        output = self.mlp(x_emb)
        return torch.clamp(output, -50, 50)
        
    def _progressive_emb(self, x_emb: torch.Tensor, t_frac: float) -> torch.Tensor:
        a = torch.ones(x_emb.shape[1], device=x_emb.device)
        start = int(t_frac * x_emb.shape[1] + 3)
        end = int(t_frac * x_emb.shape[1] + 4)
        
        if start < x_emb.shape[1]:
            a[start:end] = (t_frac * x_emb.shape[1]) - int(t_frac * x_emb.shape[1])
        if end < x_emb.shape[1]:
            a[end:] = 0

        return x_emb * a.unsqueeze(dim=0)


# 在原有模型字典中添加稳定版本
STABLE_MODELS = {
    "stable_fod_nesh": StableFod_NeSH,
    "stable_von_g": StableVon_G,
}
