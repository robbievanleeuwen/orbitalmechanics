from solar_system import SolarSystem
from solar_time import UniversalTime

ss = SolarSystem()

ut = UniversalTime(year=2020, month=3, day=14, hour=0, minute=0, second=0)

ss.plot_planets(ut, [0, 1, 2, 3])
