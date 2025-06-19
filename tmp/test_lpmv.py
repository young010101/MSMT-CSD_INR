import numpy as np
from scipy.special import lpmv, legendre
import matplotlib.pyplot as plt


def explain_legendre_polynomials():
    """解释勒让德多项式的层次关系"""

    print("=== 勒让德多项式家族 ===")
    print("1. 普通勒让德多项式 P_l(x)")
    print("2. 关联勒让德多项式 P_l^m(x)")
    print("3. 球谐函数 Y_l^m(θ,φ) 使用 P_l^m(cos θ)")

    x = np.linspace(-1, 1, 1000)

    # 普通勒让德多项式（m=0的情况）
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 第一组：普通勒让德多项式
    ax1 = axes[0, 0]
    for l in range(4):
        P_l = legendre(l)  # 生成多项式对象
        y = P_l(x)
        ax1.plot(x, y, label=f'P_{l}(x)')
    ax1.set_title('普通勒让德多项式 P_l(x)')
    ax1.legend()
    ax1.grid(True)

    # 第二组：关联勒让德多项式，固定l=2
    ax2 = axes[0, 1]
    l = 2
    for m in range(l + 1):
        y = lpmv(m, l, x)
        ax2.plot(x, y, label=f'P_{l}^{m}(x)')
    ax2.set_title(f'关联勒让德多项式 P_{l}^m(x)')
    ax2.legend()
    ax2.grid(True)

    # 第三组：关联勒让德多项式，固定l=3
    ax3 = axes[1, 0]
    l = 3
    for m in range(l + 1):
        y = lpmv(m, l, x)
        ax3.plot(x, y, label=f'P_{l}^{m}(x)')
    ax3.set_title(f'关联勒让德多项式 P_{l}^m(x)')
    ax3.legend()
    ax3.grid(True)

    # 第四组：不同l的P_l^1(x)
    ax4 = axes[1, 1]
    m = 1
    for l in range(1, 5):
        y = lpmv(m, l, x)
        ax4.plot(x, y, label=f'P_{l}^{m}(x)')
    ax4.set_title(f'不同l的 P_l^{m}(x)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


# 手动实现关联勒让德多项式
def my_lpmv(m, l, x):
    """
    手动实现关联勒让德多项式
    使用递推关系方法
    """
    x = np.asarray(x, dtype=float)

    # 处理边界情况
    if m > l or l < 0 or m < 0:
        return np.zeros_like(x)

    # P_m^m(x) 的起始值
    if m == l:
        # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
        double_factorial = 1
        for i in range(1, 2 * m, 2):
            double_factorial *= i

        result = ((-1) ** m) * double_factorial * ((1 - x ** 2) ** (m / 2))
        return result

    # 使用递推关系计算
    # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    # P_{l}^m(x) = [x*(2l-1)*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x)] / (l-m)

    if l == m + 1:
        P_mm = my_lpmv(m, m, x)
        return x * (2 * m + 1) * P_mm

    # 递推计算
    P_mm = my_lpmv(m, m, x)
    P_m1m = my_lpmv(m, m + 1, x)

    for i in range(m + 2, l + 1):
        P_new = (x * (2 * i - 1) * P_m1m - (i + m - 1) * P_mm) / (i - m)
        P_mm = P_m1m
        P_m1m = P_new

    return P_m1m


def rodrigues_formula_demo():
    """演示罗德里格斯公式推导"""
    print("=== 罗德里格斯公式推导 ===")

    # 普通勒让德多项式的罗德里格斯公式
    # P_l(x) = (1/(2^l * l!)) * d^l/dx^l [(x^2-1)^l]

    from sympy import symbols, diff, expand, simplify, factorial

    x = symbols('x')

    print("普通勒让德多项式 P_l(x) 的前几项:")
    for l in range(4):
        # 罗德里格斯公式
        expr = (x ** 2 - 1) ** l
        for i in range(l):
            expr = diff(expr, x)
        P_l = simplify(expr / (2 ** l * factorial(l)))
        print(f"P_{l}(x) = {P_l}")

    print("\n关联勒让德多项式 P_l^m(x) = (1-x^2)^(m/2) * d^m/dx^m [P_l(x)]")


def test_comparison():
    """对比我们的实现和scipy的实现"""
    x = np.linspace(-0.99, 0.99, 100)  # 避免边界奇点

    print("=== 实现对比 ===")

    test_cases = [(1, 2), (2, 3), (0, 3), (3, 3)]

    for m, l in test_cases:
        scipy_result = lpmv(m, l, x)
        my_result = my_lpmv(m, l, x)

        # 只在有效范围内比较
        valid_mask = np.isfinite(scipy_result) & np.isfinite(my_result)

        if np.any(valid_mask):
            diff = np.abs(scipy_result[valid_mask] - my_result[valid_mask])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"P_{l}^{m}: 最大差异={max_diff:.2e}, 平均差异={mean_diff:.2e}")


def physical_meaning():
    """解释物理意义"""
    print("=== 物理意义 ===")
    print("1. 量子力学：原子轨道的角度部分")
    print("   - s轨道：l=0, m=0")
    print("   - p轨道：l=1, m=-1,0,1")
    print("   - d轨道：l=2, m=-2,-1,0,1,2")

    print("\n2. 电磁学：多极展开")
    print("   - 单极：l=0")
    print("   - 偶极：l=1")
    print("   - 四极：l=2")

    print("\n3. 地球物理：重力场/磁场建模")
    print("   - 球谐系数用于描述地球重力场异常")


# 特殊性质展示
def special_properties():
    """展示关联勒让德多项式的特殊性质"""
    x = np.linspace(-1, 1, 1000)

    print("=== 特殊性质 ===")

    # 1. 正交性
    print("1. 正交性：∫_{-1}^{1} P_l^m(x) P_{l'}^m(x) dx = 0 (l ≠ l')")

    # 2. 递推关系
    print("2. 递推关系：")
    print("   (l-m)P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m")

    # 3. 边界行为
    print("3. 边界行为：")
    l, m = 2, 1
    y = lpmv(m, l, x)
    print(f"   P_{l}^{m}(±1) = {y[0]:.6f}, {y[-1]:.6f}")

    # 4. 奇偶性
    print("4. 奇偶性：")
    print(f"   P_{l}^{m}(-x) = (-1)^(l+m) P_{l}^{m}(x)")

    # 验证奇偶性
    x_test = 0.5
    left = lpmv(m, l, -x_test)
    right = ((-1) ** (l + m)) * lpmv(m, l, x_test)
    print(f"   验证：P_{l}^{m}(-{x_test}) = {left:.6f}")
    print(f"        (-1)^{l + m} P_{l}^{m}({x_test}) = {right:.6f}")


if __name__ == "__main__":
    explain_legendre_polynomials()
    rodrigues_formula_demo()
    test_comparison()
    physical_meaning()
    special_properties()
