import numpy as np
from scipy.special import lpmv, sph_harm
import matplotlib.pyplot as plt


def my_sph_harm(m, l, theta, phi):
    """
    手动实现球谐函数

    参数:
    m: 磁量子数
    l: 角量子数
    theta: 极角数组
    phi: 方位角数组
    """
    # 计算归一化常数
    from math import factorial, sqrt, pi

    # 防止阶乘计算溢出，使用scipy的阶乘
    def safe_factorial(n):
        if n < 0:
            return 0
        return factorial(n)

    # 归一化常数
    norm = sqrt((2 * l + 1) / (4 * pi) *
                safe_factorial(l - abs(m)) / safe_factorial(l + abs(m)))

    # 关联勒让德多项式
    # 注意：scipy的lpmv参数顺序是(m, l, x)
    plm = lpmv(abs(m), l, np.cos(theta))

    # 指数项
    exp_term = np.exp(1j * m * phi)

    # 处理负m的情况
    if m < 0:
        result = norm * plm * exp_term * (-1) ** abs(m)
    else:
        result = norm * plm * exp_term

    return result


# 测试对比
def test_comparison():
    """对比我们的实现和scipy的实现"""
    l, m = 2, 1
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)

    THETA, PHI = np.meshgrid(theta, phi)

    # scipy实现
    Y_scipy = sph_harm(m, l, PHI, THETA)

    # 我们的实现
    Y_mine = my_sph_harm(m, l, THETA, PHI)

    # 比较差异
    diff = np.abs(Y_scipy - Y_mine)
    print(f"最大差异: {np.max(diff)}")
    print(f"平均差异: {np.mean(diff)}")

    return Y_scipy, Y_mine


# 可视化球谐函数
def visualize_spherical_harmonic(l, m):
    """可视化球谐函数"""
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    # 计算球谐函数
    Y = sph_harm(m, l, PHI, THETA)

    # 转换为笛卡尔坐标进行3D绘图
    X = np.sin(THETA) * np.cos(PHI)
    Y_cart = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # 使用实部的绝对值作为半径
    R = np.abs(Y.real)

    fig = plt.figure(figsize=(12, 4))

    # 实部
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X * R, Y_cart * R, Z * R,
                     facecolors=plt.cm.seismic(Y.real / np.max(np.abs(Y.real))))
    ax1.set_title(f'Real part Y_{l}^{m}')

    # 虚部
    ax2 = fig.add_subplot(132, projection='3d')
    R_imag = np.abs(Y.imag)
    ax2.plot_surface(X * R_imag, Y_cart * R_imag, Z * R_imag,
                     facecolors=plt.cm.seismic(Y.imag / np.max(np.abs(Y.imag))))
    ax2.set_title(f'Imaginary part Y_{l}^{m}')

    # 模长
    ax3 = fig.add_subplot(133, projection='3d')
    R_abs = np.abs(Y)
    ax3.plot_surface(X * R_abs, Y_cart * R_abs, Z * R_abs,
                     facecolors=plt.cm.viridis(R_abs / np.max(R_abs)))
    ax3.set_title(f'|Y_{l}^{m}|')

    plt.tight_layout()
    plt.show()


# 运行测试
if __name__ == "__main__":
    print("=== 测试球谐函数实现 ===")
    Y_scipy, Y_mine = test_comparison()

    print("\n=== 可视化球谐函数 ===")
    visualize_spherical_harmonic(2, 1)

    # 展示一些常见的球谐函数值
    print("\n=== 一些特殊值 ===")
    print(f"Y_0^0(0,0) = {sph_harm(0, 0, 0, 0)}")  # 应该是 1/sqrt(4π)
    print(f"Y_1^0(π/2,0) = {sph_harm(0, 1, 0, np.pi / 2)}")
