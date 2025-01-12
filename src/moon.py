import math
from datetime import datetime

class Moon:
    # Astronomical constants
    EPOCH = 2444238.5        # 1980 January 0.0

    # Constants defining the Sun's apparent orbit
    ELONGE = 278.833540      # ecliptic longitude of the Sun at epoch 1980.0
    ELONGP = 282.596403      # ecliptic longitude of the Sun at perigee
    ECCENT = 0.016718        # eccentricity of Earth's orbit
    SUNSMAX = 1.495985e8     # semi-major axis of Earth's orbit, km
    SUNANGSIZ = 0.533128     # sun's angular size, degrees, at semi-major axis distance

    # Elements of the Moon's orbit, epoch 1980.0
    MMLONG = 64.975464       # moon's mean longitude at the epoch
    MMLONGP = 349.383063     # mean longitude of the perigee at the epoch
    MLNODE = 151.950429      # mean longitude of the node at the epoch
    MINC = 5.145396          # inclination of the Moon's orbit
    MECC = 0.054900          # eccentricity of the Moon's orbit
    MANGSIZ = 0.5181         # moon's angular size at distance a from Earth
    MSMAX = 384401.0         # semi-major axis of Moon's orbit in km
    MPARALLAX = 0.9507       # parallax at distance a from Earth
    SYNMONTH = 29.53058868   # synodic month (new Moon to new Moon)

    @staticmethod
    def fixangle(x):
        """Fix angle to range 0-360 degrees"""
        return x - 360.0 * math.floor(x / 360.0)

    @staticmethod
    def torad(x):
        """Convert degrees to radians"""
        return x * math.pi / 180.0

    @staticmethod
    def todeg(x):
        """Convert radians to degrees"""
        return x * 180.0 / math.pi

    @staticmethod
    def jtime(t):
        """Convert Unix timestamp to Julian date"""
        return (t / 86400.0) + 2440587.5

    @staticmethod
    def kepler(m, ecc):
        """Solve Kepler's equation"""
        EPSILON = 1e-6

        m = Moon.torad(m)
        e = m
        delta = 1
        while abs(delta) > EPSILON:
            delta = e - ecc * math.sin(e) - m
            e -= delta / (1 - ecc * math.cos(e))
        return e

    def phase(self, year, month, day, hour=0, minutes=0, seconds=0):
        """Calculate moon phase for given date and time

        Args:
            year (int): Year
            month (int): Month (1-12)
            day (int): Day of month
            hour (int, optional): Hour (0-23). Defaults to 0.
            minutes (int, optional): Minutes (0-59). Defaults to 0.
            seconds (int, optional): Seconds (0-59). Defaults to 0.

        Returns:
            tuple: (
                phase: float (0-1),
                age: float (days),
                distance: float (km),
                angular_diameter: float (degrees),
                sun_distance: float (km),
                sun_angular_diameter: float (degrees),
                phase_fraction: float (0-1)
            )
        """
        # Calculate Julian date
        date_sec = datetime(year, month, day, hour, minutes, seconds).timestamp()
        pdate = self.jtime(date_sec)

        day = pdate - self.EPOCH

        # Calculation of the Sun's position
        n = self.fixangle((360 / 365.2422) * day)
        m = self.fixangle(n + self.ELONGE - self.ELONGP)

        # Solve equation of Kepler
        ec = self.kepler(m, self.ECCENT)
        ec = math.sqrt((1 + self.ECCENT) / (1 - self.ECCENT)) * math.tan(ec / 2)
        ec = 2 * self.todeg(math.atan(ec))

        # True anomaly and Lambda Sun
        lambdasun = self.fixangle(ec + self.ELONGP)

        # Orbital distance factor and distance to Sun
        f = ((1 + self.ECCENT * math.cos(self.torad(ec))) / (1 - self.ECCENT * self.ECCENT))
        sun_dist = self.SUNSMAX / f
        sun_ang = f * self.SUNANGSIZ

        # Calculation of the Moon's position
        ml = self.fixangle(13.1763966 * day + self.MMLONG)
        mm = self.fixangle(ml - 0.1114041 * day - self.MMLONGP)
        mn = self.fixangle(self.MLNODE - 0.0529539 * day)

        # Evection
        ev = 1.2739 * math.sin(self.torad(2 * (ml - lambdasun) - mm))

        # Annual equation
        ae = 0.1858 * math.sin(self.torad(m))

        # Correction term
        a3 = 0.37 * math.sin(self.torad(m))

        # Corrected anomaly
        mmp = mm + ev - ae - a3

        # Correction for the equation of the centre
        mec = 6.2886 * math.sin(self.torad(mmp))

        # Another correction term
        a4 = 0.214 * math.sin(self.torad(2 * mmp))

        # Corrected longitude
        lp = ml + ev + mec - ae + a4

        # Variation
        v = 0.6583 * math.sin(self.torad(2 * (lp - lambdasun)))

        # True longitude
        lpp = lp + v

        # Corrected longitude of the node
        np = mn - 0.16 * math.sin(self.torad(m))

        # Inclination coordinates
        y = math.sin(self.torad(lpp - np)) * math.cos(self.torad(self.MINC))
        x = math.cos(self.torad(lpp - np))

        # Ecliptic longitude and latitude
        lambdamoon = self.todeg(math.atan2(y, x))
        lambdamoon += np

        # Age of the Moon in degrees and phase
        moon_age = lpp - lambdasun
        moon_phase = (1 - math.cos(self.torad(moon_age))) / 2

        # Calculate distance of moon from the centre of the Earth
        moon_dist = (self.MSMAX * (1 - self.MECC * self.MECC)) / \
                    (1 + self.MECC * math.cos(self.torad(mmp + mec)))

        # Calculate Moon's angular diameter
        moon_diam = self.MANGSIZ / (moon_dist / self.MSMAX)

        # Calculate phase fraction
        moon_phase_fraction = self.fixangle(moon_age) / 360.0

        return (
            moon_phase,                  # illuminated fraction
            self.SYNMONTH * moon_phase_fraction,  # age of moon in days
            moon_dist,                   # distance in kilometers
            moon_diam,                   # angular diameter in degrees
            sun_dist,                    # distance to Sun
            sun_ang,                     # sun's angular diameter
            moon_phase_fraction          # phase fraction
        )

    def get_phase_name(self, phase_fraction):
        """Get the name of the moon phase based on its fraction

        Args:
            phase_fraction (float): Phase fraction from 0 to 1

        Returns:
            str: Name of the moon phase
        """
        phase = phase_fraction * 100

        if phase < 6.25:
            return "New Moon"
        elif phase < 43.75:
            return "Waxing Crescent"
        elif phase < 56.25:
            return "First Quarter"
        elif phase < 93.75:
            return "Waxing Gibbous"
        elif phase < 96.25:
            return "Full Moon"
        elif phase < 143.75:
            return "Waning Gibbous"
        elif phase < 156.25:
            return "Last Quarter"
        elif phase < 193.75:
            return "Waning Crescent"
        else:
            return "New Moon"

# Example usage
if __name__ == "__main__":
    moon = Moon()

    # Get current date
    now = datetime.now()

    # Calculate moon phase
    phase_data = moon.phase(now.year, now.month, now.day,
                            now.hour, now.minute, now.second)

    # Print results
    print(f"Date: {now}")
    print(f"Moon Phase: {phase_data[0]:.3f}")
    print(f"Moon Age (days): {phase_data[1]:.1f}")
    print(f"Moon Distance (km): {phase_data[2]:.0f}")
    print(f"Moon Angular Diameter (deg): {phase_data[3]:.4f}")
    print(f"Sun Distance (km): {phase_data[4]:.0f}")
    print(f"Sun Angular Diameter (deg): {phase_data[5]:.4f}")
    print(f"Phase Name: {moon.get_phase_name(phase_data[6])}")