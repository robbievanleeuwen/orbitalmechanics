import numpy as np
from scipy.integrate import ode
from plotly_plots import LinePlot3D, EarthPlot3D

# CONSTANTS
G = 6.67408e-20  # km^3 kg^-1 s^-2

class Body:
    """Class for a body in 3D space."""

    def __init__(self, r0, v0, mass, radius=0, name=''):
        """Init Body class.

        :param r0: Initial position of the body in the global xyz axis [km]
        :type r0: list[x, y, z]
        :param v0: Initial velocity of the body in the global xyz axis [km]
        :type v0: list[vx, vy, vz]
        :param float mass: Mass of the body [kg]
        :param float radius: Radius of the body [km] (for plotting purposes only)
        :param string name: Body name
        """

        self.r0 = r0
        self.v0 = v0
        self.mass = mass
        self.radius = radius
        self.name = name

        # store the result history
        self.t_s = []
        self.r_s = []
        self.v_s = []


class Earth(Body):
    """Class for planet Earth."""

    def __init__(self):
        """Init Earth class."""

        r0 = [0, 0, 0]
        v0 = [0, 0, 0]
        mass = 5.972e24  # kg
        radius = 6378  # km

        super().__init__(r0, v0, mass, radius, name='Earth')


class TwoBodySystem:
    """Class for a two body system in 3D space.

    In this two body system, r1 and r2 are defined as the position vectors of body1 and body2 to
    the centre of mass respectively. We solve for r2 and compute the centre of mass in order to
    determine r1. As a result, the state variable relates to r2 and consists of
    [r2x, r2y, r2z, v2x, v2y, v2z].

    As a consequence of this, if a 'zero mass' object is used it should be assigned to body2.
    """

    def __init__(self, body1, body2):
        """Init TwoBodySystem class."""

        self.body1 = body1
        self.body2 = body2

        # calculate the mu value
        m1 = self.body1.mass
        m2 = self.body2.mass
        self.mu = (m1 / (m1 + m2)) ** 3 * G * (m1 + m2)

        # calculate the initial centre of mass and velocity
        total_mass = m1 + m2
        r1 = np.array(self.body1.r0)
        r2 = np.array(self.body2.r0)
        v1 = np.array(self.body1.v0)
        v2 = np.array(self.body2.v0)
        self.comr0 = (m1 * r1 + m2 * r2) / total_mass
        self.comv0 = (m1 * v1 + m2 * v2) / total_mass

    def centre_of_mass(self, t):
        """Calculate the centre of mass of the two bodies with respect to the global xyz axis."""

        return self.comr0 + self.comv0 * t

    def save_results(self, t, state):
        """Store the time, position and velocity for each body."""

        # calculate centre of mass
        com_r = self.centre_of_mass(t)

        # calculate the mass ratio
        m1 = self.body1.mass
        m2 = self.body2.mass
        m2m1 = m2 / m1

        # extract position & velocity vectors relative to the centre of mass
        r2 = np.array(state[0:3])
        v2 = np.array(state[3:6])
        r1 = -m2m1 * r2
        v1 = -m2m1 * v2

        # store body 1 position, velocity & time
        self.body1.t_s.append(t)
        self.body1.r_s.append(r1 - com_r)
        self.body1.v_s.append(v1 - self.comv0)

        # store body 2 position, velocity & time
        self.body2.t_s.append(t)
        self.body2.r_s.append(r2 - com_r)
        self.body2.v_s.append(v2 - self.comv0)

    def derivatives(self, t, y):
        """Calculate the derivative of the position and velocity vectors."""

        # unpack state
        [rx, ry, rz, vx, vy, vz] = y
        r = np.array([rx, ry, rz])
        r_norm = np.linalg.norm(r)

        ax, ay, az = -r * self.mu / r_norm ** 3

        dydt = [vx, vy, vz, ax, ay, az]
        return dydt

    def solve(self, t_span, dt):
        """Solve the two body system from t=[0, t_span] using a time step of dt."""

        n_steps = int(np.ceil(t_span / dt) + 1)

        # initial state conditions
        y0 = np.zeros(6)
        y0[0:3] = self.body2.r0 - self.comr0
        y0[3:6] = self.body2.v0 - self.comv0
        step = 1
        self.save_results(t=0, state=y0)  # store intial conditions

        # initialise solver
        solver = ode(self.derivatives)
        solver.set_integrator('dopri5')
        solver.set_initial_value(y0, 0)

        # propagate solution
        while solver.successful() and step < n_steps:
            solver.integrate(solver.t + dt)
            step += 1

            # save solution to bodies
            self.save_results(solver.t, solver.y)

    def static_plot(self):
        """Generate a static plot of the motion of both bodies."""

        # initialise lists
        r1_s = np.array(self.body1.r_s)
        r2_s = np.array(self.body2.r_s)
        v1_s = np.array(self.body1.v_s)
        v2_s = np.array(self.body2.v_s)

        x = [r1_s[:, 0], r2_s[:, 0]]
        y = [r1_s[:, 1], r2_s[:, 1]]
        z = [r1_s[:, 2], r2_s[:, 2]]
        vx = [v1_s[:, 0], v2_s[:, 0]]
        vy = [v1_s[:, 1], v2_s[:, 1]]
        vz = [v1_s[:, 2], v2_s[:, 2]]
        t = [self.body1.t_s, self.body2.t_s]

        plot_3d = LinePlot3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, t=t)
        plot_3d.static_plot()

    def static_earth_plot(self):
        """Generate a static plot of the motion of body2 around the earth (body1)."""

        earth_plot_3d = EarthPlot3D(satellite=self.body2, earth_body=self.body1)
        earth_plot_3d.static_plot()

    def animated_earth_plot(self):
        """Generate an animated plot of the motion of body2 around the earth (body1)."""

        earth_plot_3d = EarthPlot3D(satellite=self.body2, earth_body=self.body1)
        earth_plot_3d.animated_plot()


if __name__ == "__main__":
    # create bodies
    earth = Earth()
    r0 = [earth.radius + 408, 0, 0]
    v0 = [
        0,
        np.sqrt(G * earth.mass / r0[0]) * np.cos(0.9005899),
        np.sqrt(G * earth.mass / r0[0]) * np.sin(0.9005899)
    ]
    T = 2 * np.pi / np.sqrt(G * earth.mass) * r0[0] ** 1.5
    satellite = Body(r0=r0, v0=v0, mass=100, radius=0.01, name='Satellite')

    # create two body system and solve
    two_body_system = TwoBodySystem(earth, satellite)
    two_body_system.solve(t_span=T, dt=0.01 * T)
    two_body_system.animated_earth_plot()
