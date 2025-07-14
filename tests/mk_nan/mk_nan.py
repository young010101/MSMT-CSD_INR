import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class SimpleNaNNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=45):
        super().__init__()

        # 故意不使用权重初始化，使用默认的随机初始化
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # ReLU容易导致梯度爆炸
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 故意设置一些大的初始权重
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # 使用较大的权重初始化，容易导致梯度爆炸
                nn.init.normal_(layer.weight, mean=0, std=2.0)
                nn.init.normal_(layer.bias, mean=0, std=1.0)

    def forward(self, x):
        output = self.layers(x)

        # 添加一些容易产生NaN的运算
        # 1. 大数值的指数运算
        output = torch.exp(output * 0.1)

        # 2. 除法运算（可能除以接近0的数）
        denominator = torch.abs(output) + 1e-8
        output = output / denominator

        # 3. 开方运算
        output = torch.sqrt(torch.abs(output) + 1e-8)

        return output


def create_dummy_data(n_samples=1000, input_dim=3, output_dim=45):
    """创建虚拟数据"""
    # 创建随机输入坐标
    x = torch.randn(n_samples, input_dim)
    # 创建随机目标值
    y = torch.randn(n_samples, output_dim)
    return x, y


def train_nan_network():
    """训练会产生NaN的网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建网络
    model = SimpleNaNNetwork()
    model.to(device)

    # 故意使用很高的学习率
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # 学习率很高
    criterion = nn.MSELoss()

    # 创建数据
    x_train, y_train = create_dummy_data()
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("开始训练，监控NaN出现...")

    for epoch in range(100):
        epoch_losses = []
        nan_detected = False

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # 前向传播
            output = model(x_batch)

            # 检查输出是否包含NaN
            if torch.isnan(output).any():
                print(f"🚨 NaN检测到! Epoch {epoch}, Batch {batch_idx}")
                print(f"输出统计: min={output.min():.6f}, max={output.max():.6f}")
                print(f"NaN数量: {torch.isnan(output).sum()}")
                nan_detected = True
                break

            # 计算损失
            loss = criterion(output, y_batch)

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"🚨 损失为NaN! Epoch {epoch}, Batch {batch_idx}")
                nan_detected = True
                break

            # 反向传播
            loss.backward()

            # 检查梯度是否为NaN
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"🚨 梯度NaN检测到在参数 {name}")
                    has_nan_grad = True

            if has_nan_grad:
                nan_detected = True
                break

            optimizer.step()
            epoch_losses.append(loss.item())

        if nan_detected:
            print("训练因NaN而停止")
            break

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}: 平均损失 = {avg_loss:.6f}")

        # 检查损失是否变得非常大
        if avg_loss > 1e6:
            print(f"⚠️ 损失过大 ({avg_loss:.2e})，可能即将出现NaN")


if __name__ == "__main__":
    train_nan_network()
