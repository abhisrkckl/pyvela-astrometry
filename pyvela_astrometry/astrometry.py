from functools import cached_property
from pathlib import Path
from typing import IO, Union
from astropy.coordinates import SkyCoord, angles, get_body_barycentric
from astropy.time import Time
import astropy.units as u
import numpy as np
import pint.utils


class AstrometryData:
    """A class to hold the data needed for astrometric fits
    Allows reading from/writing to pmpar.in files
    """

    @u.quantity_input
    def __init__(
        self,
        positions: SkyCoord,
        posepoch: Time,
        ra_errs: u.Quantity[u.arcsec],
        dec_errs: u.Quantity[u.arcsec],
    ):
        """
        Astrometry object

        Parameters
        ----------
        posns : astropy.coordinates.SkyCoord
        posepoch : astropy.time.Time
        ra_errs : astropy.unit.Quantity
        dec_errs : astropy.unit.Quantity

        """
        self.positions = positions
        self.posepoch = posepoch
        self.ra_errs = (
            ra_errs * np.ones(len(positions)) if ra_errs.isscalar else ra_errs
        )
        self.dec_errs = (
            dec_errs * np.ones(len(positions)) if dec_errs.isscalar else dec_errs
        )

    @cached_property
    def ra(self) -> angles.core.Longitude:
        return self.positions.ra.si

    @cached_property
    def dec(self) -> angles.core.Latitude:
        return self.positions.dec.si

    @cached_property
    def t(self) -> Time:
        return self.positions.obstime

    @cached_property
    def dt(self) -> u.Quantity:
        return (self.t - self.posepoch).to(u.s)

    @cached_property
    def ssb_earth_pos(self) -> u.Quantity:
        return get_body_barycentric("earth", self.t, "DE440").get_xyz().si.T

    @classmethod
    def from_pmpar(cls, filename: Union[str, Path, IO]):
        """Create an Astrometry object from a pmpar.in file

        Parameters
        ----------
        filename : str or file-like
        """
        mjds = []
        ras = []
        decs = []
        ra_errs = []
        dec_errs = []
        with pint.utils.open_or_use(filename) as f:
            for line in f:
                if line.startswith("epoch"):
                    posepoch = Time(float(line.split()[-1]), format="mjd")
                elif not line.startswith("#") and len(line.split()) > 0:
                    mjds.append(float(line.split()[0]))
                    ras.append(line.split()[1])
                    # this will be in seconds of time
                    ra_errs.append(float(line.split()[2]))
                    decs.append(line.split()[3])
                    dec_errs.append(float(line.split()[4]))
        mjds = Time(mjds, format="mjd")
        posns = SkyCoord(ras, decs, unit=("hour", "deg"), frame="icrs", obstime=mjds)
        # convert to proper arcsec
        ra_errs = np.array(ra_errs) * np.cos(posns.dec) * 15 * u.arcsec
        dec_errs = np.array(dec_errs) * u.arcsec
        return cls(posns, posepoch, ra_errs, dec_errs)

    def as_pmpar(self) -> str:
        """Output coordinates as pmpar.in format

        Returns
        -------
        str
        """
        s = f"epoch = {self.posepoch.mjd}\n\n"
        for i in range(len(self.posns)):
            s += f"{self.posns.obstime[i].mjd} {self.posns.ra[i].to_string(sep=':',unit=u.hourangle,precision=7)} {self.ra_errs[i].to_value(u.hourangle)*3600} {self.posns.dec[i].to_string(sep=':',precision=6)} {self.dec_errs[i].to_value(u.arcsec)}\n"

        return s


def sky_position(
    dt: u.Quantity,
    ra0: u.Quantity,
    dec0: u.Quantity,
    pmra: u.Quantity,
    pmdec: u.Quantity,
    px: u.Quantity,
    ssb_earth_pos: u.Quantity,
):
    sinα0 = np.sin(ra0)
    cosα0 = np.cos(ra0)
    sinδ0 = np.sin(dec0)
    cosδ0 = np.cos(dec0)

    x0 = np.array((cosα0 * cosδ0, sinα0 * cosδ0, sinδ0))

    xdot_ra = np.array((-sinα0, cosα0, 0)) * pmra
    xdot_dec = (-cosα0 * sinδ0, -sinα0 * sinδ0, cosδ0) * pmdec
    delta_x_pm = (xdot_ra + xdot_dec) * dt

    x1 = x0 + delta_x_pm.to_value(u.dimensionless_unscaled, equivalencies=u.dimensionless_angles())
    x1_mag = np.sqrt(np.dot(x1, x1))
    x1hat = x1 / x1_mag

    D = (u.AU / px).to(u.km, equivalencies=u.dimensionless_angles())
    r = ssb_earth_pos
    delta_x_px = (r - np.dot(r, x1hat) * x1hat) / D

    x2 = x1hat + delta_x_px
    x2_mag = np.sqrt(np.dot(x2, x2))
    x2hat = x2 / x2_mag

    dec = np.arcsin(x2hat[2])
    ra = np.arctan2(x2hat[1], x2hat[0])

    return ra, dec


def sky_positions(
    dts: u.Quantity,
    ra0: u.Quantity,
    dec0: u.Quantity,
    pmra: u.Quantity,
    pmdec: u.Quantity,
    px: u.Quantity,
    ssb_earth_poss: u.Quantity,
):
    ras = []
    decs = []
    for dt, ssb_earth_pos in zip(dts, ssb_earth_poss):
        ra, dec = sky_position(dt, ra0, dec0, pmra, pmdec, px, ssb_earth_pos)
        ras.append(ra)
        decs.append(dec)

    return u.Quantity(ras), u.Quantity(decs)


def get_lnlike(data: AstrometryData):
    def lnlike(params: np.array):
        ra0 = params[0] * u.hourangle
        dec0 = params[1] * u.degree
        pmra = params[2] * (u.mas / u.year)
        pmdec = params[3] * (u.mas / u.year)
        px = params[4] * u.mas

        ra, dec = sky_positions(data.dt, ra0, dec0, pmra, pmdec, px, data.ssb_earth_pos)

        chi2 = np.sum(((data.ra - ra) / data.ra_errs) ** 2) + np.sum(
            ((data.dec - dec) / data.dec_errs) ** 2
        )
        norm = np.sum(np.log((data.ra_errs * data.dec_errs).value))

        return -0.5 * chi2 - norm

    return lnlike
