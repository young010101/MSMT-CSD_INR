# %%
import torch

tmp_x = torch.randn(1, 3, 224, 224)
tmp_y = torch.randn(1, 3, 224, 224)
tmp_res = torch.einsum("bind,bind->bind", tmp_x, tmp_y)

# %%
import numpy as np
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = "browser"

# ????
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# ???? r(?, ?)
r = 1 + 0.3 * np.sin(theta) * np.cos(phi)

# ????????
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# ???????
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=r, colorscale='Viridis')])
fig.update_layout(title='???????', autosize=True)
fig.show()
