from planet import Planet


class SolarSystem:
    """Class for a solar system."""

    def __init__(self):
        """Inits the SolarSystem class."""

        self.planets = []

        for i in range(9):
            self.planets.append(Planet(planet_id=i))
