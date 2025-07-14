import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings


class ModelHealthMonitor:
    def __init__(self, window_size=20):
        self.window_size = window_size

        # 历史记录
        self.output_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.gradient_history = deque(maxlen=window_size)

        # 健康指标
        self.health_scores = deque(maxlen=window_size)
        self.warnings_count = 0

    def analyze_model_output(self, model_output: torch.Tensor, step: int) -> Dict:
        """深入分析模型输出健康状况"""

        with torch.no_grad():
            stats = {
                'step': step,
                'min': model_output.min().item(),
                'max': model_output.max().item(),
                'mean': model_output.mean().item(),
                'std': model_output.std().item(),
                'abs_max': model_output.abs().max().item(),
                'has_nan': torch.isnan(model_output).any().item(),
                'has_inf': torch.isinf(model_output).any().item(),
                'shape': list(model_output.shape),
            }

            # 计算分布健康度
            stats.update(self._analyze_distribution(model_output))

            # 计算健康评分 (0-100)
            health_score = self._calculate_health_score(stats)
            stats['health_score'] = health_score

            # 生成警告
            warnings = self._generate_warnings(stats)
            stats['warnings'] = warnings

            # 保存历史
            self.output_history.append(stats)
            self.health_scores.append(health_score)

            return stats

    def _analyze_distribution(self, tensor: torch.Tensor) -> Dict:
        """分析张量分布特征"""
        flat = tensor.flatten()

        # 按参数分析 (假设是Von_G的4个参数)
        if tensor.shape[-1] == 4:
            param_stats = {}
            param_names = ['f_r', 'adi', 'Dh', 'f_csf']
            for i, name in enumerate(param_names):
                param_data = tensor[:, i]
                param_stats[f'{name}_mean'] = param_data.mean().item()
                param_stats[f'{name}_std'] = param_data.std().item()
                param_stats[f'{name}_min'] = param_data.min().item()
                param_stats[f'{name}_max'] = param_data.max().item()

            return {
                'param_stats': param_stats,
                'param_balance': self._check_parameter_balance(tensor),
            }

        return {
            'percentile_25': torch.quantile(flat, 0.25).item(),
            'percentile_75': torch.quantile(flat, 0.75).item(),
            'skewness': self._calculate_skewness(flat),
        }

    def _check_parameter_balance(self, tensor: torch.Tensor) -> Dict:
        """检查4个参数的平衡性"""
        if tensor.shape[-1] != 4:
            return {}

        # 应用sigmoid后的参数分析
        sigmoid_params = torch.sigmoid(tensor)

        balance = {}
        param_names = ['f_r', 'adi', 'Dh', 'f_csf']

        for i, name in enumerate(param_names):
            param_sigmoid = sigmoid_params[:, i]

            # 检查是否过于集中在边界
            near_zero = (param_sigmoid < 0.1).float().mean().item()
            near_one = (param_sigmoid > 0.9).float().mean().item()

            balance[f'{name}_near_zero_pct'] = near_zero * 100
            balance[f'{name}_near_one_pct'] = near_one * 100
            balance[f'{name}_saturation_risk'] = max(near_zero, near_one)

        return balance

    def _calculate_health_score(self, stats: Dict) -> float:
        """计算0-100的健康评分"""
        score = 100.0

        # 严重问题 (-50分)
        if stats['has_nan'] or stats['has_inf']:
            score -= 50

        # 数值范围问题
        if stats['abs_max'] > 10:
            score -= 20
        elif stats['abs_max'] > 5:
            score -= 10

        # 标准差问题
        if stats['std'] > 5:
            score -= 15
        elif stats['std'] > 2:
            score -= 5

        # 参数平衡性 (如果有参数统计)
        if 'param_balance' in stats:
            for key, value in stats['param_balance'].items():
                if 'saturation_risk' in key and value > 0.8:
                    score -= 10  # 参数饱和风险

        # 趋势分析
        if len(self.health_scores) >= 3:
            recent_scores = list(self.health_scores)[-3:]
            if all(recent_scores[i] > recent_scores[i + 1] for i in range(len(recent_scores) - 1)):
                score -= 5  # 持续下降趋势

        return max(0, min(100, score))

    def _generate_warnings(self, stats: Dict) -> List[str]:
        """生成具体的警告信息"""
        warnings = []

        if stats['has_nan']:
            warnings.append("🚨 严重：检测到NaN值")

        if stats['has_inf']:
            warnings.append("🚨 严重：检测到Inf值")

        if stats['abs_max'] > 8:
            warnings.append(f"⚠️ 警告：输出极值过大 {stats['abs_max']:.2f}")

        if stats['std'] > 3:
            warnings.append(f"⚠️ 警告：输出方差过大 {stats['std']:.2f}")

        # 参数特定警告
        if 'param_balance' in stats:
            for key, value in stats['param_balance'].items():
                if 'saturation_risk' in key and value > 0.7:
                    param_name = key.split('_')[0]
                    warnings.append(f"⚠️ 参数{param_name}有饱和风险: {value * 100:.1f}%")

        # 趋势警告
        if len(self.health_scores) >= 5:
            recent_scores = list(self.health_scores)[-5:]
            avg_recent = np.mean(recent_scores)
            if avg_recent < 70:
                warnings.append(f"⚠️ 模型健康度持续偏低: {avg_recent:.1f}")

        return warnings

    def _calculate_skewness(self, tensor: torch.Tensor) -> float:
        """计算偏度"""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0
        return ((tensor - mean) ** 3).mean() / (std ** 3)

    def print_health_report(self, stats: Dict):
        """打印详细的健康报告"""
        print(f"\n📊 模型健康监控报告 - Step {stats['step']}")
        print("=" * 50)

        # 总体健康评分
        score = stats['health_score']
        if score >= 80:
            status = "🟢 优秀"
        elif score >= 60:
            status = "🟡 良好"
        elif score >= 40:
            status = "🟠 注意"
        else:
            status = "🔴 危险"

        print(f"总体健康评分: {score:.1f}/100 {status}")

        # 基础统计
        print(f"\n📈 输出统计:")
        print(f"  范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  均值: {stats['mean']:.3f}")
        print(f"  标准差: {stats['std']:.3f}")
        print(f"  最大绝对值: {stats['abs_max']:.3f}")

        # 参数分析
        if 'param_stats' in stats:
            print(f"\n🎯 参数分析:")
            param_names = ['f_r', 'adi', 'Dh', 'f_csf']
            for name in param_names:
                mean_key = f'{name}_mean'
                std_key = f'{name}_std'
                if mean_key in stats['param_stats']:
                    mean_val = stats['param_stats'][mean_key]
                    std_val = stats['param_stats'][std_key]
                    print(f"  {name}: 均值={mean_val:.3f}, 标准差={std_val:.3f}")

        # 警告信息
        if stats['warnings']:
            print(f"\n⚠️ 警告信息:")
            for warning in stats['warnings']:
                print(f"  {warning}")
        else:
            print(f"\n✅ 无警告")

        print("=" * 50)


# 全局健康监控实例
health_monitor = ModelHealthMonitor()
