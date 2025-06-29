from datetime import datetime
import numpy as np
import scipy.io as io
import glob, os
from astropy.coordinates import Angle
#from astropy import units as u
import matplotlib.pyplot as plt
from pathlib import Path

from ..general.env import get_env
from ..general.time import interpDt

class HorizonsData:
    def __init__(self, fname, path=None):
        '''data_name must be like venus_XXXX, mars_XXXX, ...
           The table data contains columns of 1, 10, 13, 19, 20, 23, 24, and 44
           in the table settings of Horizons sysmtem.
        '''

        self.fname = fname
        if path is None:
            self.fpath = os.path.join(get_env('hsk_horizons_data_loc'), self.fname)
        else:
            self.fpath = os.path.join(path, self.fname)
        self._find_seline()
        print('Reading lines from', self.sline+1, 'to', self.eline-1)

        dtype = [('U11'), ('U5'), ('int'), ('int'), ('float'), ('int'), ('int'), ('float'), ('f8'), ('f8'), ('f8'), ('f8'), ('f8'), ('f8'), ('f8'), ('U2'), ('f8'), ('float')]
        data = np.genfromtxt(self.fpath, dtype=dtype,
                                  skip_header=self.sline, max_rows=self.eline-self.sline-1)

        nrow = data.shape[0]
        self.timeDt = np.array([datetime.strptime(data[i][0]+data[i][1], '%Y-%b-%d%H:%M') for i in np.arange(nrow)])
        self.ra = [Angle((data[i][2], data[i][3],data[i][4]),unit='hourangle') for i in np.arange(nrow)]
        self.dec = [Angle((data[i][5], data[i][6],data[i][7]),unit='degree') for i in np.arange(nrow)]
        self.illu = np.array([data[i][8] for i in np.arange(nrow)])
        self.appdia = np.array([data[i][9] for i in np.arange(nrow)])
        self.st_dist = np.array([data[i][10] for i in np.arange(nrow)])
        self.et_dist = np.array([data[i][12] for i in np.arange(nrow)])
        self.sot_angle = np.array([data[i][14] for i in np.arange(nrow)])
        #self.sot_dir = np.array([data[i][15] for i in np.arange(nrow)])
        self.sot_dir = np.array([1 if data[i][15] == '/L' else -1 for i in np.arange(nrow)])
        self.sto_angle = np.array([data[i][16] for i in np.arange(nrow)])
        self.Ls = np.array([data[i][17] for i in np.arange(nrow)])

    def _find_seline(self):
        sestr = ['$$SOE','$$EOE']
        sstr = '$$SOE'
        estr = '$$EOE'
        seline = []
        with open(self.fpath, 'r') as f:
            data = f.readlines()
            self.nrow = len(data)

        for i,idat in enumerate(data):
            if idat.rstrip('\n') in sestr:
                seline.append(i)

        self.sline = seline[0] + 1
        self.eline = seline[1] + 1
        self.max_row = self.eline - self.sline - 1

    def interpDt(self, Dt):
        """
        The method oritinally used in the Horizons class.
        Need to chack if this method works in this class too.
        """
        self.illu_intp = interpDt(Dt, self.timeDt, self.illu)
        self.appdia_intp = interpDt(Dt, self.timeDt, self.appdia)
        self.st_dist_intp = interpDt(Dt, self.timeDt, self.st_dist)
        self.et_dist_intp = interpDt(Dt, self.timeDt, self.et_dist)
        self.sot_angle_intp = interpDt(Dt, self.timeDt, self.sot_angle)
        #import pdb; pdb.set_trace()
        self.sot_dir_intp = interpDt(Dt, self.timeDt, self.sot_dir)
        self.sto_angle_intp = interpDt(Dt, self.timeDt, self.sto_angle)
        self.Ls_intp = interpDt(Dt, self.timeDt, self.Ls)

class PlanetView:
    def __init__(self, planet_name, date, slit=None):
        self.name = planet_name
        self.date = date
        self.slit = slit
        self.timeDt = datetime.strptime(date, '%Y%m%d')
        parent = Path(__file__).resolve().parent
        path = parent.joinpath('horizons_data')
        eph = HorizonsData(self.name+'_all_daily.txt', path=path)
        eph.interpDt(self.timeDt)
        self.appdia = eph.appdia_intp
        if eph.sot_dir_intp >= 0:
            theta = eph.sto_angle_intp - 90.
            phi = np.arange(181) + 90.
        elif eph.sot_dir_intp < 0:
            theta = -(eph.sto_angle_intp - 90.)
            phi = np.arange(181) + 90.

        self.y_terminator = (self.appdia/2.)*np.sin(np.radians(theta))*np.cos(np.radians(phi))
        self.z_terminator = (self.appdia/2.)*np.sin(np.radians(phi))

    def plot(self, ax=None, xlim=[-40, 40], ylim=[-40, 40]):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        circle1 = plt.Circle((0, 0), self.appdia/2, fill=False, linewidth=1)
        ax.add_patch(circle1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.plot(self.y_terminator, self.z_terminator, 'k', linewidth=1)
        ax.set_ylabel('[arcsec]')
        ax.set_xlabel('[arcsec]')
        ax.set_aspect('equal', adjustable='box')

        if self.slit is not None:
            if self.slit == 10:
                ax.axhline(5, linestyle='--', linewidth=1, color='skyblue')
                ax.axhline(-5, linestyle='--', linewidth=1, color='skyblue')
            if self.slit == 60:
                ax.axhline(30, linestyle='--', linewidth=1, color='skyblue')
                ax.axhline(-30, linestyle='--', linewidth=1, color='skyblue')

def plot_venus_from_earth(date, slit=None, ax=None, xlim=[-40, 40], ylim=[-40, 40]):
    pv = PlanetView('venus', date, slit)
    pv.plot(ax=ax, xlim=xlim, ylim=ylim)

def plot_mars_from_earth(date, slit=None, ax=None, xlim=[-15, 15], ylim=[-15, 15]):
    pv = PlanetView('mars', date, slit)
    pv.plot(ax=ax, xlim=xlim, ylim=ylim)
