import numpy as np


def kepler_E(e, M):
    """Solve Kepler's equation for the eccentric anomaly.

    Based on algorithm 3.1 on page 115.

    :param float e: Eccentricity
    :param float M: Mean anomaly (radians)

    :returns: Eccentric anomaly (radians)
    :rtype: float
    """

    tol = 1.0e-8  # set an error tolerance

    # select a starting value for E
    if M < np.pi:
        E = M + e / 2
    else:
        E = M - e / 2

    error = 1

    while abs(error) > tol:
        error = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - error

    return E


def universal_anomaly(r0_norm, vr0, delta_t, mu, alpha):
    """Calculate the univeral anomaly using Newton's method.

    Based on algorithm 3.3 on page 138.

    :param float r0_norm: Initial distance
    :param float vr0: Intial radial component of velocity
    :param float delta_t: Time elapsed since the intial conditions
    :param float mu: Gravitational parameter
    :param float alpha: Reciprocal of the semimajor axis

    :returns: The universal anomaly
    :rtype: float
    """

    chi = np.sqrt(mu) * np.abs(alpha) * delta_t  # initial estimate for the universal anomaly
    tol = 1e-8  # convergence tolerance
    max_iter = 1000  # iteration limit
    i = 0  # step counter
    error = 1  # error tracker

    while np.abs(error) > tol and i < max_iter:
        # calculate the stumpff functions
        (C, S) = stumpff_functions(alpha, chi)

        # calculate algorithm function and its derivative
        f = (
            r0_norm * vr0 / np.sqrt(mu) * chi * chi * C
            + (1 - alpha * r0_norm) * chi ** 3 * S
            + r0_norm * chi
            - np.sqrt(mu) * delta_t
        )
        dfdchi = (
            r0_norm * vr0 / np.sqrt(mu) * chi * (1 - alpha * chi * chi * S)
            + (1 - alpha * r0_norm) * chi * chi * C
            + r0_norm
        )

        error = f / dfdchi  # calculate error
        chi = chi - error  # update chi
        i += 1  # update step counter
        print(S)

    if i >= max_iter:
        print("Max iterations exceeded in universal_anomaly algorithm.")

    return chi


def orbital_update(r0, v0, delta_t, mu):
    """Calculate the orbital state given the initial state vector and a time difference.

    Based on algorithm 3.4 on page 142.

    :param r0: Initial position vector
    :type r0: :class:`numpy.ndarray`
    :param v0: Initial velocity vector
    :type v0: :class:`numpy.ndarray`
    :param float delta_t: Time elapsed since the intial conditions
    :param float mu: Gravitational parameter

    :returns: Current state and velocity vector (r, v)
    :rtype: tuple(:class:`numpy.ndarray`)
    """

    r0_norm = np.sqrt(np.dot(r0, r0))  # calculate the initial distance
    v0_norm = np.sqrt(np.dot(v0, v0))  # calculate the initial speed
    vr0 = np.dot(r0, v0) / r0_norm  # calculate the intial radial component of velocity

    alpha = 2 / r0_norm - v0_norm * v0_norm / mu  # calculate the reciprocal of the semimajor axis
    # note if alpha > 0 orbit is an ellipse; if alpha = 0 orbit is a parabola;
    # if alpha < 0 orbit is a hyperbola

    chi = universal_anomaly(r0_norm, vr0, delta_t, mu, alpha)  # calculate the univeral anomaly
    (C, S) = stumpff_functions(alpha, chi)  # calculate the stumpff functions

    # calculate the lagrange position coefficients
    f = 1 - chi * chi / r0_norm * C
    g = delta_t - 1 / np.sqrt(mu) * chi ** 3 * S

    r = f * r0 + g * v0  # calculate the new position vector
    r_norm = np.sqrt(np.dot(r, r))  # calculate the initial distance

    # calculate the lagrange position coefficients
    fdot = np.sqrt(mu) / (r_norm * r0_norm) * (alpha * chi ** 3 * S - chi)
    gdot = 1 - chi * chi / r_norm * C

    v = fdot * r0 + gdot * v0  # calculate the new velocity vector

    return (r, v)


def orbital_elements(r, v, mu, deg=False):
    """Calculate the orbital elements given the state vector.

    Based on algorithm 4.1 on page 159.

    The orbital elements are as follows:
    * h: Magnitude of the specific angular momentum
    * i: Inclincation
    * Omega: Right acension of the ascending node
    * e: Eccentricity
    * omega: Argument of perigee
    * theta: True anomaly
    * a: Semi-major axis

    :param r: Position vector
    :type r: :class:`numpy.ndarray`
    :param v: Velocity vector
    :type v: :class:`numpy.ndarray`
    :param float mu: Gravitational parameter
    :param bool deg: Whether or not to output in degrees or radians

    :returns: Returns the orbital elements (h, i, Omega, e, omega, theta, a)
    :rtype: tuple(float)
    """

    r_norm = np.sqrt(np.dot(r, r))  # calculate the distance
    v_norm = np.sqrt(np.dot(v, v))  # calculate the speed

    v_r = np.dot(r, v) / r_norm  # calculate the radial velocity
    # note if v_r > 0, the satellite is flying away from perigee
    # if vr < 0, it is flying towards perigee

    h = np.cross(r, v)  # calculate the specific angular momentum
    h_norm = np.sqrt(np.dot(h, h))  # calculate the magnitude of the specific angular momentum

    print(h, h_norm)

    i = np.arccos(h[2] / h_norm)  # calculate the inclination
    # note if the 0 < i < pi/2 the orbit is prograge
    # if pi/2 < i < pi the orbit is retrograde

    N = np.cross([0, 0, 1], h)  # calculate the node line
    N_norm = np.sqrt(np.dot(N, N))  # calculate the magnitude of the node line

    # calculate the right acension of the ascending node
    if N[1] >= 0:
        Omega = np.arccos(N[0] / N_norm)
    else:
        Omega = 2 * np.pi - np.arccos(N[0] / N_norm)

    # calculate the eccentricity vector
    e = 1 / mu * ((v_norm * v_norm - mu / r_norm) * r - r_norm * v_r * v)
    e_norm = np.sqrt(np.dot(e, e))  # calculate the eccentricity

    # calculate the argument of perigee
    if e[2] >= 0:
        omega = np.arccos(np.dot(N, e) / (N_norm * e_norm))
    else:
        omega = 2 * np.pi - np.arccos(np.dot(N, e) / (N_norm * e_norm))

    # calculate the true anomaly
    if v_r >= 0:
        theta = np.arccos(np.dot(e, r) / (e_norm * r_norm))
    else:
        theta = 2 * np.pi - np.arccos(np.dot(e, r) / (e_norm * r_norm))

    # calculate the semi-major axis
    a = h_norm * h_norm / mu / (1 - e_norm * e_norm)

    if deg:
        return (
            h_norm,
            i * 180 / np.pi,
            Omega * 180 / np.pi,
            e_norm,
            omega * 180 / np.pi,
            theta * 180 / np.pi,
            a,
        )
    else:
        return (h_norm, i, Omega, e_norm, omega, theta, a)


def state_vector(h, i, Omega, e, omega, theta, mu):
    """Calculate the state vector given the six orbital elements.

    Based on algorithm 4.2 on page 175.

    :param float h: Magnitude of the specific angular momentum
    :param float i: Inclincation
    :param float Omega: Right acension of the ascending node
    :param float e: Eccentricity
    :param float omega: Argument of perigee
    :param float theta: True anomaly
    :param float mu: Gravitational parameter

    :returns: Current state and velocity vector (r, v)
    :rtype: tuple(:class:`numpy.ndarray`)
    """

    r_xbar = h * h / (mu * (1 + e * np.cos(theta))) * np.array([np.cos(theta), np.sin(theta), 0])
    v_xbar = mu / h * np.array([-np.sin(theta), e + np.cos(theta), 0])

    a = np.cos(Omega) * np.cos(omega) - np.sin(Omega) * np.sin(omega) * np.cos(i)
    b = -np.cos(Omega) * np.sin(omega) - np.sin(Omega) * np.cos(i) * np.cos(omega)
    c = np.sin(Omega) * np.sin(i)
    d = np.sin(Omega) * np.cos(omega) + np.cos(Omega) * np.cos(i) * np.sin(omega)
    e = -np.sin(Omega) * np.sin(omega) + np.cos(Omega) * np.cos(i) * np.cos(omega)
    f = -np.cos(Omega) * np.sin(i)
    g = np.sin(i) * np.sin(omega)
    j = np.sin(i) * np.cos(omega)
    k = np.cos(i)

    Q_xbarX = np.array([[a, b, c], [d, e, f], [g, j, k]])
    print(Q_xbarX)

    r_X = np.matmul(Q_xbarX, r_xbar)
    v_X = np.matmul(Q_xbarX, v_xbar)

    return (r_X, v_X)


def lambert(r1, r2, delta_t, mu):
    """Calculate the velocity vectors given two positions and the time difference between them.

    Assumes a prograde trajectory (0 < i < 90).

    Based on algorithm 5.2 on page 208.

    :param r1: Position vector 1
    :type r1: :class:`numpy.ndarray`
    :param r2: Position vector 2
    :type r2: :class:`numpy.ndarray`
    :param float delta_t: Time difference between the two position vectors
    :param float mu: Gravitational parameter

    :returns: Velocity vectors v1 & v2
    :rtype: tuple(:class:`numpy.ndarray`)
    """

    r1_norm = np.sqrt(np.dot(r1, r1))  # calculate the distance to r1
    r2_norm = np.sqrt(np.dot(r2, r2))  # calculate the distance to r2

    if np.cross(r1, r2)[2] >= 0:
        delta_theta = np.arccos(np.dot(r1, r2) / (r1_norm * r2_norm))
    else:
        delta_theta = 2 * np.pi - np.arccos(np.dot(r1, r2) / (r1_norm * r2_norm))

    A = np.sin(delta_theta) * np.sqrt(r1_norm * r2_norm / (1 - np.cos(delta_theta)))

    def y(z, C, S):
        return r1_norm + r2_norm + A * (z * S - 1) / (np.sqrt(C))

    # solve eq. 5.39 for z using Newton's method
    tol = 1e-8  # convergence tolerance
    max_iter = 5000  # iteration limit
    i = 0  # step counter
    error = 1  # error tracker
    z = 0  # initial guess for z

    while np.abs(error) > tol and i < max_iter:
        # calculate algorithm function and its derivative
        (C, S) = stumpff_functions(z, 1)
        y_z = y(z, C, S)
        f = (y_z / C) ** 1.5 * S + A * np.sqrt(y_z) - np.sqrt(mu) * delta_t

        if z == 0:
            dfdz = np.sqrt(2) / 40 * y_z ** 1.5 + A / 8 * (
                np.sqrt(y_z) + A * np.sqrt(1 / (2 * y_z))
            )
        else:
            dfdz = (y_z / C) ** 1.5 * (
                1 / (2 * z) * (C - 1.5 * S / C) + 0.75 * S * S / C
            ) + A / 8 * (3 * S / C * np.sqrt(y_z) + A * np.sqrt(C / y_z))

        error = f / dfdz  # calculate error
        z = z - error  # update chi
        i += 1  # update step counter

    if i >= max_iter:
        print("Max iterations exceeded in lambert algorithm.")

    y_z = y_z = y(z, C, S)  # calculate final value of y

    # determine lagrange coefficients
    f = 1 - y_z / r1_norm
    g = A * np.sqrt(y_z / mu)
    g_dot = 1 - y_z / r2_norm

    # calculate velocities
    v1 = 1 / g * (r2 - f * r1)
    v2 = 1 / g * (g_dot * r2 - r1)

    return (v1, v2)


def julian(y, m, d):
    """Calculate julian day number at 0 UT between 1900 & 2100.

    Based on Equation 5.48 on page 214.

    :param int y: Year between 1900 & 2100, i.e. 1901 <= y <= 2099.
    :param int m: Month i.e. 1 <= m <= 12
    :param int d: Day i.e. 1 <= m <= 31

    :returns: Julian day number
    :rtype: float
    """

    return (
        367 * y - np.fix(7 * (y + np.fix((m + 9) / 12)) / 4) + np.fix(275 * m / 9) + d + 1721013.5
    )


def planetry_elements(planet_id):
    """Extract a planet's J2000 orbital elements and centennial rates.

    Based on Table 8.1 on page 388.

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

    :returns: J2000 orbital elements and centennial rates.
    :rtype: tuple(:class:`numpy.ndarray`)
    """

    J2000_elements = np.array(
        [
            [0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084],
            [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
            [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
            [1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084, 355.45332],
            [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
            [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
            [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
            [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
            [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881],
        ]
    )

    cent_rates = np.array(
        [
            [0.00000066, 0.00002527, -23.51, -446.30, 573.57, 538101628.29],
            [0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
            [-0.00000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
            [-0.00007221, 0.00011902, -25.47, -1020.19, 1560.78, 68905103.78],
            [0.00060737, -0.00012880, -4.15, 1217.17, 839.93, 10925078.35],
            [-0.00301530, -0.00036762, 6.11, -1591.05, -1948.89, 4401052.95],
            [0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
            [-0.00125196, 0.00002514, -3.64, -151.25, -844.43, 786449.21],
            [-0.00076912, 0.00006465, 11.07, -37.33, -132.25, 522747.90],
        ]
    )

    J2000_oe = J2000_elements[planet_id, :]
    rates = cent_rates[planet_id, :]

    # convert AU to km
    au = 149597871
    J2000_oe[0] *= au
    rates[0] *= au

    # convert arcseconds to fractions of a degree
    rates[2:] *= 1 / 3600

    return (J2000_oe, rates)


def planet_state(planet_id, y, m, d, hour, minute, second, deg=False):
    """Calculate the orbital elements and state vector of a planet from the date and UT.

    Based on algorithm 8.1 on page 389.

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

    The orbital elements are as follows:
    * h: Magnitude of the specific angular momentum
    * i: Inclincation
    * Omega: Right acension of the ascending node
    * e: Eccentricity
    * omega: Argument of perigee
    * theta: True anomaly
    * a: Semi-major axis

    :param int planet_id: ID of the planet
    :param int y: Year between 1900 & 2100, i.e. 1901 <= y <= 2099.
    :param int m: Month i.e. 1 <= m <= 12
    :param int d: Day i.e. 1 <= m <= 31
    :param int hour: UT hour i.e. 0 <= hour <= 23
    :param int minute: UT minute i.e. 0 <= minute <= 59
    :param int second: UT second i.e. 0 <= second <= 59
    :param bool deg: Whether or not to output in degrees or radians

    :returns: Orbital elements (h, i, Omega, e, omega, theta, a, omega_hat, L, M, E), the state
        vector (r, v) and the Julian day
    :rtype: tuple(tuple(float), :class:`numpy.ndarray`, :class:`numpy.ndarray`, float)
    """

    mu = 1.327124e11  # mu for planets orbiting the sun
    j0 = julian(y, m, d)  # julian day at UT 0
    ut = (hour + minute / 60 + second / 3600) / 24  # univeral time
    jd = j0 + ut  # julian day
    t0 = (jd - 2451545) / 36525  # calculate number of julian centuries between J2000 and date

    (J2000_oe, rates) = planetry_elements(planet_id)  # get data for the planet
    orbital_elements = J2000_oe + rates * t0  # calculate orbital elements at date

    # extract orbital elements
    a = orbital_elements[0]
    e = orbital_elements[1]
    h = np.sqrt(mu * a * (1 - e * e))
    i = orbital_elements[2]
    Omega = zero_to_360(orbital_elements[3])
    omega_hat = zero_to_360(orbital_elements[4])
    L = zero_to_360(orbital_elements[5])
    omega = zero_to_360(omega_hat - Omega)

    # calculate true anomaly
    M = zero_to_360((L - omega_hat))
    E = kepler_E(e, M * np.pi / 180)
    theta = zero_to_360(2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)) * 180 / np.pi)

    # collect the orbital elements (degrees)
    oe = (h, i, Omega, e, omega, theta, a, omega_hat, L, M, E * 180 / np.pi)

    # calculate the state vector (angles in radians)
    (r, v) = state_vector(
        h, i * np.pi / 180, Omega * np.pi / 180, e, omega * np.pi / 180, theta * np.pi / 180, mu
    )

    if deg:
        return (oe, r, v, jd)
    else:
        angle_list = [1, 2, 4, 5, 7, 8, 9, 10]
        oe = list(oe)

        for i in angle_list:
            oe[i] *= np.pi / 180

        return (oe, r, v, jd)


def interplanetary(depart, arrive):
    """Calculate spacecraft trajectory from SOI of planet 1 to planet 2.

    Based on algorithm 8.2 on page 393.

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

    :param depart: List of parameters defining the departure
        [planet_id, y, m, d, hour, minute, second]
    :type depart: list(float)
    :param arrive: List of parameters defining the arrival
        [planet_id, y, m, d, hour, minute, second]
    :type arrive: list(float)

    :returns: The state vector and time of depature at planet 1 (rp1, vp1, jd1), the state vector
        and time of arrival at planet 2 (rp2, vp2, jd2) and the velocity at departure and arrival
        (v1, v2).
    :rtype: tuple(tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, float),
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, float),
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`))
    """

    mu = 1.327124e11  # mu for planets orbiting the sun

    # get state vector at departure planet
    (_, rp1, vp1, jd1) = planet_state(
        depart[0], depart[1], depart[2], depart[3], depart[4], depart[5], depart[6]
    )

    # get state vector at arrival planet
    (_, rp2, vp2, jd2) = planet_state(
        arrive[0], arrive[1], arrive[2], arrive[3], arrive[4], arrive[5], arrive[6]
    )

    # calculate time of flight
    tof = (jd2 - jd1) * 24 * 3600

    # calculate trajectory by lambert's problem
    [v1, v2] = lambert(rp1, rp2, tof, mu)

    return ((rp1, vp1, jd1), (rp2, vp2, jd2), (v1, v2))


def stumpff_functions(alpha, chi):
    """Calculate the stumpff functions C & S as a function of z = alpha * chi * chi.

    :param float alpha: Reciprocal of the semimajor axis
    :param float chi: Univeral anomaly

    :returns: The stumpff functions (C, S)
    :rtype: tuple(float)
    """

    z = alpha * chi * chi  # calculate variable z

    if z > 0:
        S = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
        C = (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
        C = (np.cosh(np.sqrt(-z)) - 1) / (-z)
    elif z == 0:
        S = 1.0 / 6
        C = 1.0 / 2

    return (C, S)


if __name__ == "__main__":
    mu = 398600
    # depart = [2, 1996, 11, 7, 0, 0, 0]
    # arrive = [3, 1997, 9, 12, 0, 0, 0]

    # print(interplanetary(depart, arrive))

    # r0 = np.array([7000, -12124, 0])
    # v0 = np.array([2.6679, 4.6210, 0])
    # dt = 3600
    #
    # (r, v) = orbital_update(r0, v0, dt, mu)
    #
    # print(r, v)

    # r = np.array([-6045, -3490, 2500])
    # v = np.array([-3.457, 6.618, 2.533])
    #
    # (h_norm, i, Omega, e_norm, omega, theta, a) = orbital_elements(r, v, mu, deg=True)
    # print(h_norm, i, Omega, e_norm, omega, theta, a)

    deg = np.pi / 180
    print(state_vector(80000, 30 * deg, 40 * deg, 1.4, 60 * deg, 30 * deg, mu))
