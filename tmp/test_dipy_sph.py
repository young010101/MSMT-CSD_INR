# %%
import numpy as np

from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.sims.voxel import multi_tensor_odf
from dipy.viz import actor, window

# %%
# 我们可以首先使用球极坐标在 HemiSphere 上创建一些随机点。
rng = np.random.default_rng()
n_pts = 64
theta = np.pi * rng.random(n_pts)
phi = 2 * np.pi * rng.random(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)

# %%
# 接下来，我们调用 disperse_charges 函数，它会迭代地移动点，使静电势能最小化。
# 在 hsph_updated 中，我们得到了更新后的 HemiSphere ，其中的点在半球上分布均匀。
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
sphere = Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))

# %%
# 现在我们需要创建初始信号。为此，我们将使用球体的顶点作为球面函数 (SF) 的采样点。
# 我们将使用 multi_tensor_odf 来模拟 ODF。有关如何使用 DIPY 模拟信号和 ODF 的更多信息，请参阅 多张量模拟 。
mevals = np.array([[0.0015, 0.00015, 0.00015], [0.0015, 0.00015, 0.00015]])
angles = [(0, 0), (60, 0)]
odf = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

# Enables/disables interactive visualization
interactive = True

scene = window.Scene()
scene.SetBackground(1, 1, 1)

odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere)
odf_actor.RotateX(90)
scene.add(odf_actor)

print("Saving illustration as symm_signal.png")
window.record(scene=scene, out_path="../reports/figs/symm_signal.png", size=(300, 300))
if interactive:
    window.show(scene)

# %%
# 我们现在可以用一系列 SH 系数来表示这个信号 sf_to_sh 。
# 此函数将一系列 SF 系数转换为一系列 SH 系数。有关 SH 基的更多信息，请参阅球谐函数基 。
# 在本例中，我们将使用 descoteaux07 基，最大 SH 阶数为 8。
# Change this value to try out other bases
sh_basis = "descoteaux07"
# Change this value to try other maximum orders
sh_order_max = 8

sh_coeffs = sf_to_sh(odf, sphere, sh_order_max=sh_order_max, basis_type=sh_basis)

# %%
high_res_sph = get_sphere(name="symmetric724").subdivide(n=2)
reconst = sh_to_sf(
    sh_coeffs, high_res_sph, sh_order_max=sh_order_max, basis_type=sh_basis
)

scene.clear()
odf_actor = actor.odf_slicer(reconst[None, None, None, :], sphere=high_res_sph)
odf_actor.RotateX(90)
scene.add(odf_actor)

print("Saving output as symm_reconst.png")
window.record(scene=scene, out_path="../reports/figs/symm_reconst.png", size=(300, 300))
if interactive:
    window.show(scene)