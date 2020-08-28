import numpy as np
import plotly.graph_objects as go
from utils import kepler_E, state_vector, zero_to_360


class Planet:
    """Class for a planet in the solar system."""

    def __init__(self, planet_id):
        """Init the Planet class.

        Planet IDs are as follows:
        * 0 - Mercury
        * 1 - Venus
        * 2 - Earth
        * 3 - Mars
        * 4 - Jupiter
        * 5 - Saturn
        * 6 - Uranus
        * 7 - Neptune
        * 8 - Pluto

        :param int planet_id: ID of the planet
        """

        self.planet_id = planet_id
        self.mu = 1.327124e11  # mu for planets orbiting the sun
        self.au = 149597871  # astronomical unit

        names = [
            'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'
        ]
        colours = [
            '#808080', '#DEB887', '#0000FF', '#FF0000', '#FFA500', '#FFD700', '#AFEEEE', '#4682B4',
            '#8B4513'
        ]

        self.planet_name = names[planet_id]
        self.colour = colours[planet_id]

    def planetry_elements(self):
        """Extract a planet's J2000 orbital elements and centennial rates.

        Based on Table 8.1 on page 388.

        :returns: J2000 orbital elements and centennial rates.
        :rtype: tuple(:class:`numpy.ndarray`)
        """

        J2000_elements = np.array([
            [0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084],
            [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
            [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
            [1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084, 355.45332],
            [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
            [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
            [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
            [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
            [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881],
        ])

        cent_rates = np.array([
            [0.00000066, 0.00002527, -23.51, -446.30, 573.57, 538101628.29],
            [0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
            [-0.00000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
            [-0.00007221, 0.00011902, -25.47, -1020.19, 1560.78, 68905103.78],
            [0.00060737, -0.00012880, -4.15, 1217.17, 839.93, 10925078.35],
            [-0.00301530, -0.00036762, 6.11, -1591.05, -1948.89, 4401052.95],
            [0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
            [-0.00125196, 0.00002514, -3.64, -151.25, -844.43, 786449.21],
            [-0.00076912, 0.00006465, 11.07, -37.33, -132.25, 522747.90],
        ])

        J2000_oe = J2000_elements[self.planet_id, :]
        rates = cent_rates[self.planet_id, :]

        # convert AU to km
        J2000_oe[0] *= self.au
        rates[0] *= self.au

        # convert arcseconds to fractions of a degree
        rates[2:] *= 1 / 3600

        return (J2000_oe, rates)

    def planet_state(self, universal_time, deg=False):
        """Calculate the orbital elements and state vector of a planet from the date and UT.

        Based on algorithm 8.1 on page 389.

        The orbital elements are as follows:
        * h: Magnitude of the specific angular momentum
        * i: Inclincation
        * Omega: Right acension of the ascending node
        * e: Eccentricity
        * omega: Argument of perigee
        * theta: True anomaly
        * a: Semi-major axis
        * omega_hat: xxx
        * L: xxx
        * M: xxx
        * E: xxx

        :param universal_time: Universal time object
        :type universal_time: :class:`~orbitalmechanics.UniversalTime`
        :param bool deg: Whether or not to output in degrees or radians

        :returns: Orbital elements (h, i, Omega, e, omega, theta, a, omega_hat, L, M, E), the state
            vector (r, v) and the Julian day
        :rtype: tuple(tuple(float), :class:`numpy.ndarray`, :class:`numpy.ndarray`, float)
        """

        jd = universal_time.julian_time()
        t0 = universal_time.j2000_difference()

        (J2000_oe, rates) = self.planetry_elements()  # get data for the planet
        orbital_elements = J2000_oe + rates * t0  # calculate orbital elements at date

        # extract orbital elements
        a = orbital_elements[0]
        e = orbital_elements[1]
        h = np.sqrt(self.mu * a * (1 - e * e))
        i = orbital_elements[2]
        Omega = zero_to_360(orbital_elements[3])
        omega_hat = zero_to_360(orbital_elements[4])
        L = zero_to_360(orbital_elements[5])
        omega = zero_to_360(omega_hat - Omega)

        # calculate true anomaly
        M = zero_to_360((L - omega_hat))
        E = kepler_E(e, M * np.pi / 180)
        theta = zero_to_360(
            2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)) * 180 / np.pi
        )

        # collect the orbital elements (degrees)
        oe = (h, i, Omega, e, omega, theta, a, omega_hat, L, M, E * 180 / np.pi)

        # calculate the state vector (angles in radians)
        (r, v) = state_vector(
            h, i * np.pi / 180, Omega * np.pi / 180, e, omega * np.pi / 180, theta * np.pi / 180,
            self.mu
        )

        if deg:
            return (oe, r, v, jd)
        else:
            angle_list = [1, 2, 4, 5, 7, 8, 9, 10]
            oe = list(oe)

            for i in angle_list:
                oe[i] *= np.pi / 180

            return (oe, r, v, jd)

    def plot_orbit(self, fig, universal_time, n=60):
        """Plot the orbit and current state of the planet using plotly.

        :param fig: Plotly graph object
        :type: :class:`plotly.graph_objs._figure.Figure`
        :param universal_time: Universal time object
        :type universal_time: :class:`~orbitalmechanics.UniversalTime`
        :param int n: Number of points used to plot the orbit
        """

        (oe, r, v, jd) = self.planet_state(universal_time)  # get planet state

        # initialise orbit results
        thetas = np.linspace(0, 2 * np.pi, n + 1)
        x = np.zeros(n + 1)
        y = np.zeros(n + 1)
        z = np.zeros(n + 1)

        # plot orbit
        for (i, theta) in enumerate(thetas):
            (r_t, v_t) = state_vector(oe[0], oe[1], oe[2], oe[3], oe[4], theta, self.mu)
            x[i] = r_t[0]
            y[i] = r_t[1]
            z[i] = r_t[2]

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            name=self.planet_name,
            showlegend=False,
            mode='lines',
            line={
                'color': self.colour
            }
        ))

        # plot position
        fig.add_trace(go.Scatter3d(
            x=[r[0]],
            y=[r[1]],
            z=[r[2]],
            name=self.planet_name,
            mode='markers',
            marker={
                'color': self.colour,
                'size': 6
            }
        ))
