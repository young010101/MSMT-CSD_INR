# %%

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, disperse_charges
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.tracking.tests.test_tractogen import get_sphere
from dipy.viz import actor, window

# default_rng 的作用是：创建一个随机数生成器
rng = np.random.default_rng()

# 随机生成 n_pts 个点
n_pts = 64
theta = np.pi * rng.random(n_pts)
phi = 2 * np.pi * rng.random(n_pts)
# 半球上的点
hsph_initial = HemiSphere(theta=theta, phi=phi)

hsph_updated, potential = disperse_charges(hsph_initial, 5000)

# %% 可视化
# can ignore this
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hsph_updated.x, hsph_updated.y, hsph_updated.z)
ax.set_title("测试方向")
plt.show()


# %%
vertices = hsph_updated.vertices
values = np.ones(vertices.shape[0])

bvecs = np.vstack((vertices, vertices))
bvals = np.hstack((1000 * values, 2500 * values))

# %%
# 添加一些b0
bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
bvals = np.insert(bvals, (0, bvals.shape[0]), 0)

# %%
# 创建模拟信号的过程？
gtab = gradient_table(bvals, bvecs=bvecs)

# 1. 物理单位
# 这些值的单位是 mm²/s（平方毫米每秒），代表扩散系数：
#
# 0.0015 mm²/s = 1.5 × 10⁻³ mm²/s
# 这是真实人体组织中的典型扩散率

# 2. 生物学意义
# λ1 = 0.0015：沿纤维方向的扩散（轴向扩散率，AD）
# λ2 = λ3 = 0.00015：垂直纤维方向的扩散（径向扩散率，RD）
#
# 3. 真实范围
# 在真实的脑白质中：
#
# 轴向扩散率（AD）：0.8-1.8 × 10⁻³ mm²/s
# 径向扩散率（RD）：0.2-0.6 × 10⁻³ mm²/s
mevals = np.array([[0.001, 0.0008, 0.0003], [0.0015, 0.0003, 0.0003]])
angles = [(0, 0), (90, 0)]
fractions = [80, 20]

signal, sticks = multi_tensor(gtab, mevals, angles=angles, fractions=fractions, snr=None)

signal_noisy, _ = multi_tensor(gtab, mevals, angles=angles, fractions=fractions, snr=20)

plt.plot(signal, label='noiseless')

plt.plot(signal_noisy, label='with noise')
plt.legend()
# plt.show()
plt.savefig('test_multi_tensor_simulator.png', bbox_inches='tight')

# %%
sphere = get_sphere(name="symmetric724")
sphere = sphere.subdivide(n=2)

odf = multi_tensor_odf(sphere.vertices, mevals, angles=angles, fractions=fractions)

interactive = True

scene = window.Scene()

odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere, colormap="plasma")
odf_actor.RotateX(90)

scene.add(odf_actor)

print("Saving output as symm_reconst.png")
window.record(scene=scene, out_path="../reports/figs/symm_reconst.png", size=(300, 300))
if interactive:
    window.show(scene)
