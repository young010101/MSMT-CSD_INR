import numpy as np
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor_odf
from dipy.viz import actor, window

# # 准备数据
# ## 模拟数据
mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
# (theta, phi)
angles = [(0, 0), (60, 0), (120, 0)]
fractions = [30, 30, 40]

# ## 创建球面
# ### 返回 724 个顶点
sphere = get_sphere(name="symmetric724")
# ### 细分，获得11554 个顶点
sphere = sphere.subdivide(n=2)

# ## 模拟ODF
# ### 返回 11554 个 ODF，n_vertices 个
odf = multi_tensor_odf(sphere.vertices, mevals, angles=angles, fractions=fractions)

# vis
interactive = True

scene = window.Scene()

odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere, colormap="plasma")
# 绕X轴旋装90度
# odf_actor.RotateX(90)
# odf_actor.RotateY(90)
# odf_actor.RotateZ(90)

scene.add(odf_actor)
print("Saving output as symm_reconst.png")
window.record(scene=scene, out_path="../reports/figs/symm_reconst.png", size=(300, 300))
if interactive:
    window.show(scene)