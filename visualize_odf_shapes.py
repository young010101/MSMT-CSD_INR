import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

def create_sphere_grid(n_points=50):
    """创建球面网格"""
    phi = np.linspace(0,2*np.pi, n_points)
    theta = np.linspace(0, np.pi, n_points)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    return phi_grid, theta_grid

def reconstruct_odf(coeffs, phi, theta):
    """从球谐系数重建ODF"""
    odf = np.zeros_like(phi, dtype=complex)
    coeff_idx = 0
    for l in range(0, 9, 2):  # l = 0, 2, 4, 6, 8
        for m in range(-l, l+1):
            Y_lm = sph_harm(m, l, theta, phi)
            odf += coeffs[coeff_idx] * Y_lm
            coeff_idx += 1
    return np.real(odf)

# 文件路径
nii_files = {
    "全量训练": "runs/mwu100408_250716/coeffs_test.nii.gz",
    "70%欠采样": "runs/mwu100408_250716/coeffs_colfg8vl_random_sample.nii.gz",
    "奇偶分离": "runs/mwu100408_250716/coeffs_tcqivmvu_oddeven.nii.gz"
}# 选择要分析的像素位置
slice_id = 45
pixel_x, pixel_y = 72, 87 # 选择中心附近的像素

# 创建球面网格
phi_grid, theta_grid = create_sphere_grid(50)

# 读取数据并分析
odf_shapes = {}
for name, path in nii_files.items():
    print(f"分析 {name} 的ODF形状...")
    
    # 加载数据
    img = nib.load(path)
    data = img.get_fdata()
    
    # 提取指定像素的45个系数
    if data.ndim == 4:
        coeffs = data[pixel_x, pixel_y, slice_id, :45] # 取前45个系数
    else:
        coeffs = data[pixel_x, pixel_y, slice_id]
    
    # 重建ODF
    odf = reconstruct_odf(coeffs, phi_grid, theta_grid)
    odf_shapes[name] = odf
    
    print(f"  {name} 系数范围: [{np.min(coeffs):.4f}, {np.max(coeffs):.4f}]")
    print(f"  {name} ODF范围:[{np.min(odf):.4f}, {np.max(odf):.4}]")

# 1 3D可视化对比
print("生成3D ODF形状对比...")
fig_3d, axes_3d = plt.subplots(1,3, figsize=(18,6), subplot_kw={'projection': '3d'})

for i, (name, odf) in enumerate(odf_shapes.items()):
    x = np.sin(theta_grid) * np.cos(phi_grid) * odf
    y = np.sin(theta_grid) * np.sin(phi_grid) * odf
    z = np.cos(theta_grid) * odf
    
    surf = axes_3d[i].plot_surface(x, y, z, cmap='viridis', alpha=0.8)
    axes_3d[i].set_title(f"{name}\nODF Shape")
    axes_3d[i].set_xlabel("X")
    axes_3d[i].set_ylabel("Y")
    axes_3d[i].set_zlabel('Z')

plt.tight_layout()
plt.savefig('odf_shapes_3d_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

#2. 2投影对比
print("生成2D ODF投影对比...")
fig_2d, axes_2d = plt.subplots(3, 2, figsize=(12,15))

for i, (name, odf) in enumerate(odf_shapes.items()):
    # 极坐标投影
    ax_polar = plt.subplot(3, 2, 2*i+1, projection='polar')
    im1 = ax_polar.contourf(phi_grid, theta_grid, odf, levels=20, cmap='viridis')
    ax_polar.set_title(f'{name} - Polar Projection')
    plt.colorbar(im1, ax=ax_polar)
    
    # 笛卡尔投影
    ax_cart = plt.subplot(3, 2, 2*i+2)
    im2 = ax_cart.imshow(odf, cmap='viridis', extent=(0, 2*np.pi, 0, np.pi))
    ax_cart.set_xlabel("Phi")
    ax_cart.set_ylabel('Theta')
    ax_cart.set_title(f'{name} - Cartesian Projection')
    plt.colorbar(im2, ax=ax_cart)

plt.tight_layout()
plt.savefig('odf_shapes_2d_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

#3. 系数对比分析
print("生成系数对比分析...")
fig_coeffs, axes_coeffs = plt.subplots(1,3, figsize=(18,5))
for i, (name, path) in enumerate(nii_files.items()):
    img = nib.load(path)
    data = img.get_fdata()
    
    if data.ndim == 4:
        coeffs = data[pixel_x, pixel_y, slice_id, :45]
    else:
        coeffs = data[pixel_x, pixel_y, slice_id]
    
    axes_coeffs[i].bar(range(45), coeffs, alpha=0.7)
    axes_coeffs[i].set_title(f'{name} - ODF Coefficients')
    axes_coeffs[i].set_xlabel("Coefficient Index")
    axes_coeffs[i].set_ylabel("Coefficient Value")
    axes_coeffs[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('odf_coefficients_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

#4. 生成分析报告
print("生成ODF形状分析报告...")
with open('odf_shape_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("ODF形状分析报告\n")
    f.write("=" * 50 + "\n")
    
    f.write(f"分析位置: 像素({pixel_x}, {pixel_y}), Slice {slice_id}\n\n")
    
    for name, odf in odf_shapes.items():
        f.write(f"{name} ODF分析:\n")
        f.write(f"  - ODF最小值: {np.min(odf):0.6f}\n")
        f.write(f"  - ODF最大值: {np.max(odf):0.6f}\n")
        f.write(f"  - ODF均值: {np.mean(odf):0.6f}\n")
        f.write(f"  - ODF标准差: {np.std(odf):0.6f}\n")
        f.write(f"  - 主要方向数量: {len(np.where(odf > np.max(odf)*0.8)[0])}\n\n")

print("ODF形状分析完成！生成的文件:")
print("- odf_shapes_3d_comparison.png: 3D ODF形状对比")
print("- odf_shapes_2d_comparison.png: 2D ODF投影对比")
print("- odf_coefficients_comparison.png: ODF系数对比")
print("- odf_shape_analysis_report.txt: ODF形状分析报告") 