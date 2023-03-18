from datetime import datetime
import numpy as np
import scipy.io as io
import glob
from astropy.coordinates import Angle
from astropy import units as u
import matplotlib.pyplot as plt

import common.tools as ctools
from common.tools import interpDt
from rc import saveloc, horizonsloc


class HorizonsData:
    def __init__(self, data_name, data_loc='sc_operation'):
        '''data_name must be like venus_XXXX, mars_XXXX, ...
           The table data contains columns of 1, 10, 13, 19, 20, 23, 24, and 44
           in the table settings of Horizons sysmtem.
        '''
        self.data_name = data_name
        self.target = self.data_name.split('_')[0].capitalize()
        self.path = horizonsloc + data_loc+'/'
        self.fname = sorted(glob.glob(self.path + data_name + '.txt'))
        if ~isinstance(self.fname, str):
            self.fname = self.fname[0]
        self._find_seline()

        dtype = [('U11'), ('U5'), ('int'), ('int'), ('float'), ('int'), ('int'), ('float'), ('f8'), ('f8'), ('f8'), ('f8'), ('f8'), ('f8'), ('f8'), ('U2'), ('f8'), ('float')]
        print(self.sline, self.eline)
        data = np.genfromtxt(self.fname, dtype=dtype,
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
        self.Ls = np.array([data[i][16] for i in np.arange(nrow)])

    def _find_seline(self):
        sestr = ['$$SOE','$$EOE']
        sstr = '$$SOE'
        estr = '$$EOE'
        seline = []
        with open(self.fname, 'r') as f:
            data = f.readlines()
            self.nrow = len(data)

        for i,idat in enumerate(data):
            if idat.rstrip('\n') in sestr:
                seline.append(i)

        self.sline = seline[0] + 1
        self.eline = seline[1] + 1
        self.max_row = self.eline - self.sline - 1

    def calc_radec_sky(self,
                       dra=Angle((0, 0, 0) ,unit='degree'),
                       ddec=Angle((1, 0, 0) ,unit='degree')):
        self.ra_sky = [ira + dra for ira in self.ra]
        self.dec_sky = [idec + ddec for idec in self.dec]

    def get_date_str(self):
        date_list = [iDt.strftime('%Y%m%d') for iDt in self.timeDt]
        return date_list

    def get_ra_str(self, sky=False):
        if sky:
            return [[f'{ira.hms[0]:02.0f}', f'{ira.hms[1]:02.0f}', f'{ira.hms[2]:05.2f}'] for ira in self.ra_sky]
        else:
            return [[f'{ira.hms[0]:02.0f}', f'{ira.hms[1]:02.0f}', f'{ira.hms[2]:05.2f}'] for ira in self.ra]

    def get_dec_str(self, sky=False):
        if sky:
            return [['-'+f'{-idec.dms[0]+0:02.0f}', f'{-idec.dms[1]:02.0f}', f'{-idec.dms[2]:04.1f}'] if (idec.dms[0]<0)|(idec.dms[1]<0)|(idec.dms[2]<0)
            else ['+'+f'{idec.dms[0]+0:02.0f}', f'{idec.dms[1]:02.0f}', f'{idec.dms[2]:04.1f}' ] for idec in self.dec_sky ]
        else:
            return [['-'+f'{-idec.dms[0]+0:02.0f}', f'{-idec.dms[1]:02.0f}', f'{-idec.dms[2]:04.1f}'] if (idec.dms[0]<0)|(idec.dms[1]<0)|(idec.dms[2]<0)
            else ['+'+f'{idec.dms[0]+0:02.0f}', f'{idec.dms[1]:02.0f}', f'{idec.dms[2]:04.1f}' ] for idec in self.dec ]

    def save_radec_list(self, sky=True):
        save_name = self.path + 'out/' + self.target + '_' +'radec_list_' + datetime.now().strftime('%Y%m%d') + '.txt'
        with open(save_name, 'w') as f:
            second_column = '12345'
            name = self.target + '_sky' if sky else self.target
            date_list = self.get_date_str()
            ra_list = self.get_ra_str(sky=sky)
            dec_list = self.get_dec_str(sky=sky)

            for i in range(len(self.ra)):
                s = name + '_' + date_list[i] + ' ' + second_column + ' ' +\
                    ra_list[i][0] + ' ' + ra_list[i][1] + ' ' + ra_list[i][2] + ' ' +\
                    dec_list[i][0] + ' ' + dec_list[i][1] + ' ' + dec_list[i][2]
                f.write(s+'\n')

    def interpDt(self, Dt):
        """
        The method oritinally used in the Horizons class.
        Need to chack if this method works in this class too.
        """
        pass
        self.illu_intp = interpDt(Dt, self.timeDt, self.illu)
        self.appdia_intp = interpDt(Dt, self.timeDt, self.appdia)
        self.st_dist_intp = interpDt(Dt, self.timeDt, self.st_dist)
        self.et_dist_intp = interpDt(Dt, self.timeDt, self.et_dist)
        self.sot_angle_intp = interpDt(Dt, self.timeDt, self.sot_angle)
        #import pdb; pdb.set_trace()
        self.sot_dir_intp = interpDt(Dt, self.timeDt, self.sot_dir)
        self.sto_angle_intp = interpDt(Dt, self.timeDt, self.sto_angle)

def gen_sky_radec_list(data_name, sky=True, dra=Angle((0, 0, 0) ,unit='degree'), ddec=Angle((1, 0, 0) ,unit='degree')):
    h=HorizonsData(data_name)
    h.calc_radec_sky(dra=dra, ddec=ddec)
    h.save_radec_list(sky=sky)


def gen_tar_radec_list(data_name, sky=False):
    h=HorizonsData(data_name)
    h.save_radec_list(sky=sky)


class PlanetView:
    def __init__(self, planet_name, date, slit=None):
        self.name = planet_name
        self.date = date
        self.slit = slit
        self.timeDt = datetime.strptime(date, '%Y%m%d')
        eph = HorizonsData(self.name+'_all_daily', data_loc=self.name)
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
    vv = PlanetView('venus', date, slit)
    vv.plot(ax=ax, xlim=xlim, ylim=ylim)

def plot_mars_from_earth(date, slit=None, ax=None, xlim=[-15, 15], ylim=[-15, 15]):
    vv = PlanetView('mars', date, slit)
    vv.plot(ax=ax, xlim=xlim, ylim=ylim)
