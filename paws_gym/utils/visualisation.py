import numpy as np
import plotly.graph_objs as go
from paws_gym.motion.model import Model

t = np.linspace(0, 5, 500)
m = Model('trot')
foot_positions = [m.compute(step) for step in t]

x_trajs = [[], [], [], []]
y_trajs = [[], [], [], []]
z_trajs = [[], [], [], []]

for feet in foot_positions:
    x_trajs[0].append(0.2+feet[0][0])
    y_trajs[0].append(0.1+feet[0][1])
    z_trajs[0].append(feet[0][2])

    x_trajs[1].append(0.2+feet[1][0])
    y_trajs[1].append(-0.1+feet[1][1])
    z_trajs[1].append(feet[1][2])

    x_trajs[2].append(-0.2+feet[2][0])
    y_trajs[2].append(0.1+feet[2][1])
    z_trajs[2].append(feet[2][2])

    x_trajs[3].append(-0.2+feet[3][0])
    y_trajs[3].append(-0.1+feet[3][1])
    z_trajs[3].append(feet[3][2])


fig = go.Figure(
    layout=go.Layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title="Animated 3D Trajectories of Four Points",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 10, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])]
        )]
    )
)

for i in range(len(x_trajs)):
    fig.add_trace(go.Scatter3d(
        x=x_trajs[i],
        y=y_trajs[i],
        z=z_trajs[i],
        mode='lines+markers',
        marker=dict(size=5),
        line=dict(width=2),
    ))

frames = []
for j in range(len(foot_positions)):
    frame_data = []
    for i in range(4):
        frame_data.append(go.Scatter3d(
            x=[x_trajs[i][j]],
            y=[y_trajs[i][j]],
            z=[z_trajs[i][j]],
            mode="markers+lines",
            marker=dict(size=5),
            line=dict(width=2)
        ))
    frames.append(go.Frame(data=frame_data, name=str(j)))

fig.frames = frames

fig.show()