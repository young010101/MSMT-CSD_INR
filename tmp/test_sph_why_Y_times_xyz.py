import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D


def explain_radial_visualization():
    """详细解释径向缩放可视化的原理"""

    print("=== 球谐函数3D可视化原理 ===")
    print()
    print("1. 球谐函数本质：定义在单位球面上的函数")
    print("   Y_l^m(θ,φ) : S² → ℂ")
    print("   (θ,φ) ∈ [0,π] × [0,2π]")
    print()
    print("2. 单位球面的参数化：")
    print("   x = sin(θ)cos(φ)")
    print("   y = sin(θ)sin(φ)")
    print("   z = cos(θ)")
    print("   满足：x² + y² + z² = 1")
    print()
    print("3. 径向缩放可视化：")
    print("   新坐标 = 原坐标 × 缩放因子")
    print("   (x',y',z') = (x,y,z) × |Y_l^m(θ,φ)|")
    print()
    print("4. 几何意义：")
    print("   - |Y| > 1: 球面向外凸出")
    print("   - |Y| < 1: 球面向内凹陷")
    print("   - Y = 0: 保持在单位球面上")


def step_by_step_visualization():
    """逐步演示可视化过程"""

    # 创建球面网格
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)

    # 单位球面的笛卡尔坐标
    X = np.sin(THETA) * np.cos(PHI)
    Y_cart = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    print("步骤1: 验证单位球面")
    radius_check = np.sqrt(X ** 2 + Y_cart ** 2 + Z ** 2)
    print(f"半径检查: min={np.min(radius_check):.6f}, max={np.max(radius_check):.6f}")
    print("(应该都等于1)")

    # 计算球谐函数
    l, m = 2, 1
    Y = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y.real)  # 使用实部的绝对值作为径向因子

    print(f"\n步骤2: 球谐函数 Y_{l}^{m}")
    print(f"值域: [{np.min(Y.real):.3f}, {np.max(Y.real):.3f}]")
    print(f"径向因子 R: [{np.min(R):.3f}, {np.max(R):.3f}]")

    # 可视化对比
    fig = plt.figure(figsize=(15, 5))

    # 原始单位球面
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y_cart, Z, alpha=0.7, color='lightblue')
    ax1.set_title('步骤1: 单位球面')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 球谐函数值（颜色映射）
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y_cart, Z,
                     facecolors=plt.cm.RdBu(Y.real / np.max(np.abs(Y.real))))
    ax2.set_title(f'步骤2: Y_{l}^{m} 颜色映射')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # 径向缩放后
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X * R, Y_cart * R, Z * R,
                     facecolors=plt.cm.RdBu(Y.real / np.max(np.abs(Y.real))))
    ax3.set_title(f'步骤3: 径向缩放 Y_{l}^{m}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

    return X, Y_cart, Z, Y, R


def mathematical_justification():
    """数学证明为什么这样做是正确的"""

    print("=== 数学证明 ===")
    print()
    print("目标：在3D空间中可视化球面函数 f(θ,φ)")
    print()
    print("方法：径向变形 (radial deformation)")
    print("原始点: P(θ,φ) = (sin θ cos φ, sin θ sin φ, cos θ)")
    print("变形点: P'(θ,φ) = f(θ,φ) × P(θ,φ)")
    print()
    print("性质：")
    print("1. 保持方向：P'与P同方向（只改变距离）")
    print("2. 几何直观：函数值大的地方'凸出'更多")
    print("3. 连续变形：保持球面的拓扑结构")
    print()
    print("等价表示：")
    print("P'(θ,φ) = [f(θ,φ) sin θ cos φ, f(θ,φ) sin θ sin φ, f(θ,φ) cos θ]")
    print("       = f(θ,φ) × [sin θ cos φ, sin θ sin φ, cos θ]")
    print("       = R × [X, Y, Z]")


def alternative_visualizations():
    """展示其他可视化方法"""

    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2 * np.pi, 30)
    THETA, PHI = np.meshgrid(theta, phi)

    X = np.sin(THETA) * np.cos(PHI)
    Y_cart = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    l, m = 2, 1
    Y = sph_harm(m, l, PHI, THETA)

    fig = plt.figure(figsize=(20, 4))

    # 方法1: 径向缩放（原方法）
    ax1 = fig.add_subplot(141, projection='3d')
    R = np.abs(Y.real)
    ax1.plot_surface(X * R, Y_cart * R, Z * R,
                     facecolors=plt.cm.RdBu(Y.real / np.max(np.abs(Y.real))))
    ax1.set_title('方法1: 径向缩放')

    # 方法2: 只用颜色，不变形
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.plot_surface(X, Y_cart, Z,
                     facecolors=plt.cm.RdBu(Y.real / np.max(np.abs(Y.real))))
    ax2.set_title('方法2: 颜色映射')

    # 方法3: 符号径向缩放（保留正负号）
    ax3 = fig.add_subplot(143, projection='3d')
    R_signed = Y.real / np.max(np.abs(Y.real)) + 1  # 平移到正值
    ax3.plot_surface(X * R_signed, Y_cart * R_signed, Z * R_signed,
                     facecolors=plt.cm.RdBu(Y.real / np.max(np.abs(Y.real))))
    ax3.set_title('方法3: 符号缩放')

    # 方法4: 双层显示（正负值分离）
    ax4 = fig.add_subplot(144, projection='3d')
    R_pos = np.maximum(Y.real, 0) / np.max(np.abs(Y.real))
    R_neg = np.abs(np.minimum(Y.real, 0)) / np.max(np.abs(Y.real))

    # 正值向外
    ax4.plot_surface(X * (1 + R_pos), Y_cart * (1 + R_pos), Z * (1 + R_pos),
                     alpha=0.7, color='red')
    # 负值向内
    ax4.plot_surface(X * (1 - R_neg), Y_cart * (1 - R_neg), Z * (1 - R_neg),
                     alpha=0.7, color='blue')
    ax4.set_title('方法4: 双层显示')

    plt.tight_layout()
    plt.show()


def radius_interpretation():
    """解释径向距离的物理意义"""

    print("=== 径向距离的物理解释 ===")
    print()
    print("在扩散MRI中：")
    print("- 径向距离 ∝ 扩散概率/强度")
    print("- ' 凸出' 方向 = 主要扩散方向")
    print("- 对称性 = 扩散的各向异性模式")
    print()
    print("在量子力学中：")
    print("- 径向距离 ∝ 电子云密度")
    print("- 轨道形状直观显示")
    print()
    print("数学上：")
    print("- 这是球面函数可视化的标准方法")
    print("- 保持了函数的所有对称性")
    print("- 便于识别零点、极值、节线等特征")


# 运行演示
if __name__ == "__main__":
    explain_radial_visualization()
    print("\n" + "=" * 50 + "\n")

    X, Y_cart, Z, Y, R = step_by_step_visualization()
    print("\n" + "=" * 50 + "\n")

    mathematical_justification()
    print("\n" + "=" * 50 + "\n")

    alternative_visualizations()
    print("\n" + "=" * 50 + "\n")

    radius_interpretation()
