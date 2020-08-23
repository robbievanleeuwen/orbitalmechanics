import numpy as np
import plotly.graph_objects as go


class LinePlot2D():
    """a."""

    def __init__(self, x, y, vx, vy, t):
        """Init the LinePlot2D class.

        x,y,.. = list of list of [[x0..xn], [x0..xn], ...]
        """

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.t = t

        self.fig = go.Figure()

    def static_plot(self):
        """Generate a 2D static plot."""

        for (i, x_s) in enumerate(self.x):
            y_s = self.y[i]
            v_x = self.vx[i]
            v_y = self.vy[i]
            t = self.t[i]

            # generate labels
            text = []

            for (i, t_i) in enumerate(t):
                text.append('vx: {0:.3f}<br>vy: {1:.3f}<br>t: {2:.3f}'.format(v_x[i], v_y[i], t_i))

            self.fig.add_trace(go.Scatter(
                x=x_s,
                y=y_s,
                mode='lines+markers',
                hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>%{text}',
                name='',
                text=text))

        self.fig.update_layout(
            title={
                'text': 'Projectile Motion',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='x',
            yaxis_title='y'
        )

        self.fig.show()


class LinePlot3D():
    """a."""

    def __init__(self, x, y, z, vx, vy, vz, t):
        """Init the LinePlot3D class.

        x,y,.. = list of list of [[x0..xn], [x0..xn], ...]
        """

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.t = t

        self.fig = go.Figure()

    def static_plot(self):
        """Generate a 3D static plot."""

        for (i, t) in enumerate(self.t):
            x_s = self.x[i]
            y_s = self.y[i]
            z_s = self.z[i]
            v_x = self.vx[i]
            v_y = self.vy[i]
            v_z = self.vz[i]

            # generate labels
            text = []

            for (i, t_i) in enumerate(t):
                text.append(
                    'vx: {0:.3f}<br>vy: {1:.3f}<br>vz: {2:.3f}<br>t: {3:.3f}'.format(
                        v_x[i], v_y[i], v_z[i], t_i
                    )
                )

            self.fig.add_trace(go.Scatter3d(
                x=x_s,
                y=y_s,
                z=z_s,
                mode='lines+markers',
                hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>%{text}',
                name='',
                text=text))

        self.fig.update_layout(
            title={
                'text': 'Two Body Problem',
                'x': 0.5,
                'xanchor': 'center'
            }
        )

        self.fig.show()


class EarthPlot3D():
    """a."""

    def __init__(self, satellite, earth_body):
        """Init the EarthPlot3D class.

        x,y,.. = list of list of satellite state [x0..xn]
        """

        self.satellite = satellite
        self.earth_body = earth_body

        r_s = np.array(self.satellite.r_s)
        v_s = np.array(self.satellite.v_s)

        self.x = r_s[:, 0]
        self.y = r_s[:, 1]
        self.z = r_s[:, 2]
        self.vx = v_s[:, 0]
        self.vy = v_s[:, 1]
        self.vz = v_s[:, 2]
        self.t = self.satellite.t_s

        self.fig = go.Figure()

    def static_plot(self):
        """Generate a 3D earth static plot."""

        # generate labels
        text = []

        for (i, t_i) in enumerate(self.t):
            text.append(
                'vx: {0:.3f}<br>vy: {1:.3f}<br>vz: {2:.3f}<br>t: {3:.3f}'.format(
                    self.vx[i], self.vy[i], self.vz[i], t_i
                )
            )

        # draw satellite
        self.fig.add_trace(go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode='lines+markers',
            marker=dict(
                size=4,
                opacity=0.8
            ),
            hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>%{text}',
            name=self.satellite.name,
            text=text
        ))

        # draw earth
        radius = self.earth_body.radius
        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, np.pi, 50)
        x = radius * np.outer(np.cos(theta), np.sin(phi))
        y = radius * np.outer(np.sin(theta), np.sin(phi))
        z = radius * np.outer(np.ones(50), np.cos(phi))

        self.fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=z,
            name=self.earth_body.name,
            hoverinfo='skip',
            showscale=False
        ))

        self.fig.update_layout(
            title={
                'text': 'Earth Satellite',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1))
        )

        self.fig.show()

    def animated_plot(self):
        """Generate a 3D earth animated plot."""

        # draw earth
        radius = self.earth_body.radius
        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, np.pi, 50)
        x = radius * np.outer(np.cos(theta), np.sin(phi))
        y = radius * np.outer(np.sin(theta), np.sin(phi))
        z = radius * np.outer(np.ones(50), np.cos(phi))

        self.fig = go.Figure(
            data=[
                go.Surface(
                    x=x, y=y, z=z, name=self.earth_body.name, hoverinfo='skip', showscale=False
                ),
                go.Surface(
                    x=x, y=y, z=z, name=self.earth_body.name, hoverinfo='skip', showscale=False
                )],
            frames=[
                go.Frame(
                    data=go.Scatter3d(
                        x=self.x[0:k],
                        y=self.y[0:k],
                        z=self.z[0:k],
                        mode='lines+markers',
                        marker=dict(
                            size=4,
                            opacity=0.8
                        ),
                        name=self.satellite.name
                    ),
                    name=str(k),
                )
                for k in range(len(self.x))
            ])

        # Layout
        self.fig.update_layout(
            title={
                'text': 'Earth Satellite',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis=dict(
                    range=[-8000, 8000], autorange=False),
                yaxis=dict(
                    range=[-8000, 8000], autorange=False),
                zaxis=dict(
                    range=[-8000, 8000], autorange=False),
                aspectratio=dict(
                    x=1, y=1, z=1),
            ),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    # 'args': [None]
                    'args': [None, {"frame": {"duration": 50, "redraw": True}}]
                }]
            }]
        )

        self.fig.show()
