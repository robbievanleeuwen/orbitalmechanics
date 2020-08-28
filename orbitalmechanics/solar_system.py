import plotly.graph_objects as go
from planet import Planet


class SolarSystem:
    """Class for a solar system."""

    def __init__(self):
        """Inits the SolarSystem class."""

        self.planets = []

        for i in range(9):
            self.planets.append(Planet(planet_id=i))

    def plot_planets(self, universal_time, planet_ids=[]):
        """Plot the current state of the planets and their orbits using plotly.

        :param universal_time: Universal time object
        :type universal_time: :class:`~orbitalmechanics.UniversalTime`
        :param int planet_ids: A list of planet ids to plot, if empty plots all planets
        """

        fig = go.Figure()

        # plot the sun
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            name='The Sun',
            mode='markers',
            marker={
                'color': '#FFFF00',
                'size': 12
            }
        ))

        # plot the planets
        for planet in self.planets:
            if planet.planet_id in planet_ids or not planet_ids:
                planet.plot_orbit(fig=fig, universal_time=universal_time)

        fig.update_layout(
            title={
                'text': 'Solar System - {0}'.format(universal_time),
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1))
        )

        fig.show()
