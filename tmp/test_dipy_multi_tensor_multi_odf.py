import numpy as np
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor_odf
from dipy.viz import actor, window

# 创建多个不同的ODF
sphere = get_sphere(name="symmetric724").subdivide(n=2)

# 模拟3个不同的ODF
mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])

# 三个不同的配置
odf_configs = [
    {"angles": [(0, 0), (90, 0)], "fractions": [50, 50]},
    {"angles": [(0, 0), (60, 0), (120, 0)], "fractions": [30, 30, 40]},
    {"angles": [(45, 0), (135, 0)], "fractions": [60, 40]}
]

# 生成多个ODF
odfs = []
for config in odf_configs:
    odf = multi_tensor_odf(sphere.vertices, mevals, **config)
    odfs.append(odf)

# 将ODF排列成3D网格形状 (例如 1x3x1 的网格)
odf_volume = np.zeros((1, 3, 1, len(sphere.vertices)))
for i, odf in enumerate(odfs):
    odf_volume[0, i, 0, :] = odf

# 创建可视化
scene = window.Scene()
odf_actor = actor.odf_slicer(odf_volume, sphere=sphere, colormap="plasma")
scene.add(odf_actor)
window.show(scene)