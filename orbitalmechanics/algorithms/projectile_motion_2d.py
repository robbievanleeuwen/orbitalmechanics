import numpy as np
from plotly_plots import LinePlot2D

# PARAMETERS:
x0 = 0
y0 = 100
v0 = 50
gamma0 = np.pi / 6
g = 9.81
n = 50


def plot_2d_projectile():
    """Assumes y0 >=0."""

    # solve time of flight for y = 0
    a = -0.5 * g
    b = v0 * np.sin(gamma0)
    c = y0
    det = (b * b - 4 * a * c)
    t_roots = ((- b - np.sqrt(det)) / (2 * a), (-b + np.sqrt(det)) / (2 * a))
    t_end = max(t_roots)

    t_s = np.linspace(0, t_end, n)
    x_s = np.zeros(n)
    y_s = np.zeros(n)
    vx_s = np.zeros(n)
    vy_s = np.zeros(n)

    for (i, t) in enumerate(t_s):
        (x_s[i], y_s[i], vx_s[i], vy_s[i]) = projectile_equation_2d(t)

    plot_2d = LinePlot2D(x=[x_s], y=[y_s], vx=[vx_s], vy=[vy_s], t=[t_s])
    plot_2d.static_plot()


def projectile_equation_2d(t):
    """2D formula for projectile motion."""

    x = x0 + v0 * np.cos(gamma0) * t
    y = y0 + v0 * np.sin(gamma0) * t - 0.5 * g * t * t
    vx = v0 * np.cos(gamma0)
    vy = v0 * np.sin(gamma0) - g * t

    return (x, y, vx, vy)


if __name__ == "__main__":
    plot_2d_projectile()
