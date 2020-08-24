import numpy as np


def kepler_E(e, M):
    """Solve Kepler's equation for the eccentric anomaly.

    Based on algorithm 3.1 on page 115.

    :param float e: Eccentricity
    :param float M: Mean anomaly (radians)

    :returns: Eccentric anomaly (radians)
    :rtype: float
    """

    tol = 1.e-8  # set an error tolerance

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

    Q_xbarX = np.array([
        [a, b, c],
        [d, e, f],
        [g, j, k]]
    )

    r_X = np.matmul(Q_xbarX, r_xbar)
    v_X = np.matmul(Q_xbarX, v_xbar)

    return (r_X, v_X)


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


def zero_to_360(x):
    """Reduce an angle to lie between 0 and 360 degrees.

    :param float x: Angle

    :returns: Angle between 0 and 360 degrees.
    :rtype: float
    """

    if x >= 360:
        return x - np.fix(x / 360) * 360
    elif x < 0:
        return x - (np.fix(x / 360) - 1) * 360
    else:
        return x
