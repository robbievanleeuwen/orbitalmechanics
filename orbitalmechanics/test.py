from solar_system import SolarSystem
from solar_time import UniversalTime
import matplotlib.pyplot as plt

ss = SolarSystem()

ut = UniversalTime(year=2020, month=3, day=14, hour=0, minute=0, second=0)

x = []
y = []

for planet in ss.planets:
    (_, r, _, _) = planet.planet_state(universal_time=ut)
    print('{0} - {1}'.format(planet.planet_name, r))

    x.append(r[0])
    y.append(r[1])

plt.plot(0, 0, 'ro')
plt.plot(x, y, 'bx')
plt.show()
