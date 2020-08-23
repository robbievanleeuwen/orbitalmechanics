import numpy as np


class UniversalTime:
    """Class for a universal time."""

    def __init__(self, year, month, day, hour, minute, second):
        """Init the UniversalTime class.

        :param int year: Year between 1900 & 2100, i.e. 1901 <= y <= 2099.
        :param int month: Month i.e. 1 <= m <= 12
        :param int day: Day i.e. 1 <= m <= 31
        :param int hour: UT hour i.e. 0 <= hour <= 23
        :param int minute: UT minute i.e. 0 <= minute <= 59
        :param int second: UT second i.e. 0 <= second <= 59
        """

        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    def julian_day(self):
        """Calculate julian day number at 0 UT between 1900 & 2100.

        Based on Equation 5.48 on page 214.

        :returns: Julian day number
        :rtype: float
        """

        return (
            367 * self.year - np.fix(7 * (self.year + np.fix((self.month + 9) / 12)) / 4)
            + np.fix(275 * self.month / 9) + self.day + 1721013.5
        )

    def julian_time(self):
        """Calculate the julian day number at the given UT.

        :returns: Julian day number
        :rtype: float
        """

        return self.julian_day() + (self.hour + self.minute / 60 + self.second / 3600) / 24

    def j2000_difference(self):
        """Calculate the number of julian centuries between J2000 and the given date.

        :returns: Julian centuries
        :rtype: float
        """

        return (self.julian_time() - 2451545) / 36525
