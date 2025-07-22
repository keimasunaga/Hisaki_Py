import glob, os
import numpy as np
import astropy.io.fits as fits
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib import colors
import urllib.request
from scipy.ndimage import map_coordinates
import scipy.io as sio
from pathlib import Path

from .env import get_env
from .time import str2Dt, get_timeDt_mean
from .calib import get_cal_daily, get_cal, get_xbin_lim

# Hisaki data location
dataloc = get_env('hsk_l2_data_loc')
dataloc_l2p = get_env('hsk_l2p_data_loc')
dataloc_l3 = get_env('hsk_l3_data_loc')
url_l2p = get_env('hsk_l2p_data_url')
url_l2  = get_env('hsk_l2_data_url')
url_l2_pub  = get_env('hsk_l2_data_url_pub')
url_l3  = get_env('hsk_l3_data_url')

# Characteristic extensions
ext_primary = 0 # Primary
ext_total = 1   # Total
ext_offset = 2  # Offset

xscl = 10  # plate scale in spectral direction
yscl = 4.2 # plate scale in spatial direction

class HskData:
    '''
    HskData object opens fitsdata
    and saves some instance variables taken from filename.
    There is also a method to close the opened fitsdata.
    '''
    def __init__(self, filename, open=True):
        ## Read img data
        self.filename = filename
        self.path = dataloc + self.filename
        self.hdul = None
        name_splt = self.filename.split('.')
        self.target = name_splt[1]
        self.target_body = self.target.split('_')[0]
        self.mode = name_splt[3]
        self.date = name_splt[4]
        self.lv = name_splt[6]
        self.vr = name_splt[8]
        self.ext_offset = ext_offset
        self.ext_total = ext_total
        self.obs_period = None
        if open:
            self.open()

    def open(self):
        self.hdul = fits.open(self.path)
        print('---- Opened ' + self.filename + ' ----')
        print('---- Make sure that the fits data should be closed eventually. ----')

    def close(self):
        self.hdul.close()
        print('---- The fits data closed. ----')

    def get_nextend(self, exclude_offset=False):
        return get_nextend(self.hdul, exclude_offset)

    def get_ext_all(self):
        return get_ext_all(self.hdul)

    def get_img(self, ext=1):
        return get_img(self.hdul, ext)

    def get_header(self, ext):
        return self.hdul[ext].header

    def get_header_value(self, header_name, ext=None, fix=False):
        return get_header_value(self.hdul, header_name, ext, fix=fix)

    def get_timeDt(self, ext=None):
        return get_timeDt(self.hdul, ext)

    def get_cal(self, daily=False):
        if daily:
            self.xcal_daily, self.C2R_daily, self.C2Rtbl_daily = get_cal_daily(self.date)
            return self.xcal_daily, self.C2R_daily, self.C2Rtbl_daily
        else:
            self.xcal, self.C2R, self.C2Rtbl = get_cal()
            return self.xcal, self.C2R, self.C2Rtbl

    def get_ext_seorb(self, delta_thre=2500, add_first_and_last_exts=False):
        return get_ext_seorb(self.hdul, delta_thre, add_first_and_last_exts)

    def get_calflg(self, ext=None, string=True):
        return get_calflg(self.hdul, ext, string)

    def get_gcbc(self, ext=None):
        return get_gcbc(self.hdul, ext)

    def get_submod(self, ext=None):
        return get_submod(self.hdul, ext)

    def get_submst(self, ext=None):
        return get_submst(self.hdul, ext)

    def get_sclt(self, ext=None):
        return get_sclt(self.hdul, ext)

    def get_sclt_pla(self, ext=None):
        return get_sclt_pla(self.hdul, ext)

    def get_sclon(self, ext=None):
        return get_sclon(self.hdul, ext)

    def get_sclat(self, ext=None):
        return get_sclat(self.hdul, ext)

    def get_dist_earth_sc(self, ext=None):
        return get_dist_earth_sc(self.hdul, ext)

    def get_dist_earth_sun(self, ext=None):
        return get_dist_earth_sun(self.hdul, ext)

    def get_dist_pla_sc(self, ext=None):
        return get_dist_pla_sc(self.hdul, ext)

    def get_dist_pla_sun(self, ext=None):
        return get_dist_pla_sun(self.hdul, ext)

    def get_slit_mode(self, ext=None):
        return get_slit_mode(self.hdul, ext)


def get_fname(target, date='*', mode='*', lv='02', vr='00',
              lt='00-24', dt='00106', sky=False, # for level-2 prime data
              fullpath=False):

    if lv=='02':
        pattern = 'exeuv.'+ target + '.mod.' + mode + '.' + date + '.lv.' + lv + '.vr.' + vr + '.fits'
        filepath = glob.glob(os.path.join(dataloc, pattern))
    elif lv=='l2p':
        if sky==False:
            pattern = 'exeuv_' + target + '_' + date + '_lv02p_LT' + lt + '_dt' + dt + '_vr' + vr + '.fits'
            filepath = glob.glob(os.path.join(dataloc_l2p, target, target, date[0:4], pattern))
        else:
            pattern = 'exeuv_sky_' + date + '_lv02p_LT' + lt + '_dt' + dt + '_vr' + vr + '.fits'
            filepath = glob.glob(os.path.join(dataloc_l2p, target, 'sky', date[0:4], pattern))

    if fullpath:
        if np.size(filepath) == 1:
            filepath = filepath[0]
        if np.size(fullpath) == 0:
            print('---- No data found, returning an empty list ----')
        return filepath
    else:
        if np.size(filepath) == 1:
            fname = os.path.split(filepath[0])[1]
        else:
            fname = [os.path.split(ifname)[1] for ifname in filepath]
        if np.size(fname) == 0:
            print('---- No data found, returning an empty list ----')
        return fname

def fname2date(fname):
    if type(fname) is str:
        name_splt = fname.split('.')
        date = name_splt[4]
    if type(fname) is list:
        date = [ifname.split('.')[4] for ifname in fname]
    return date

def fname2target(fname):
    if type(fname) is str:
        name_splt = fname.split('.')
        target = name_splt[1]
    if type(fname) is list:
        target = [ifname.split('.')[1] for ifname in fname]
    return target

def fname2target_body(fname):
    if type(fname) is str:
        name_splt = fname.split('.')
        target = name_splt[1]
        target_body = target.split('_')[0]
    if type(fname) is list:
        name_splt = fname[0].split('.')
        target = name_splt[1]
        target_body = target.split('_')[0]#[ifname.split('.')[1].split('_')[0] for ifname in fname][0]
    return target_body

def fitsopen(path):
    hdul = fits.open(path)
    fname = os.path.split(path)[1] #path[0].split('/')[-1]
    print('---- Opened ' + fname + ' ----')
    print('---- Make sure that the fits data should be closed eventually. ----')
    return hdul

def fitsclose(hdul):
    hdul.close()
    print('---- The fits data closed. ----')

def get_nextend(hdul, exclude_offset=False):
    '''
    returns NEXTEND read by the primary data (ext=0).
    Note that NEXTEND does NOT count the extension of the primary data but DOES count the total data.
    When NEXTEND=n_ext, this means that fits have n_ext of spectrum data (total spectrum and each exposure spectrum) in addition to the primary data.
    Extensions thus consist of 0 (primary), 1 (total), 2, ..., n_ext.
    Therefore, when you want to read each header data expect for primary and total extensions, you have to read from ext=2 to n_ext. (i.e., list(range(2, n_ext+1)) )
    '''
    nextend = hdul[ext_primary].header['NEXTEND']
    if exclude_offset:
        nextend = nextend - 1
    return nextend

def get_ext_all(hdul):
    '''
    Returns each extension expect for primary and total., i.e., n_ext starting from 2 to n_ext+1.
    '''
    nextend = get_nextend(hdul)
    return list(range(ext_offset, nextend+1))

def get_img(hdul, ext=1):
    '''
    returns an (accumulated) image based on selected extensions (2D spectrum)
    hdul: open fits data (header data unit list)
    return: An image data (np.array([1024, 1024]))
    '''
    if isinstance(ext, (int, np.integer)):
        data = hdul[ext].data
    else:
        data = np.zeros([1024,1024])
        for i in ext:
            data += hdul[i].data
    return data

def get_header(hdul, ext):
    return hdul[ext].header

def get_header_value(hdul, header_name, ext=None, fix=False):
    '''
    Get header values for all extensions.
    Note that this does NOT include those of the primary (ext=0) and total (ext=1) extensions.
    arg:
        hdul: open fits data
        header_name: name(s) of the header (string or a list of strings)
        ext: selected extends.
             By defalt (ext=None) all extends (except for primary and total extensions (ext=0, 1)) are selected.
        fix: Hisaki data sometimes inculde nan in header values though fits data is not actually allowed to include it,
             which causes errors when reading them header values.
             This keyword fixes this problem and enebles to read the header values as a string format.
             (i.e., read as not np.nan but 'nan')
    return: An array (1 header_name) or a dictionary (>1 header_name) of header values
    '''

    if np.size(header_name) == 1:
        if fix == True:
                hdul.verify('fix')
        if ext is None:
            ext_all = get_ext_all(hdul)
            hdvalue = np.array([hdul[i].header[header_name] for i in ext_all])
        else:
            if np.size(ext) == 1:
                hdvalue = hdul[ext].header[header_name]
            else:
                hdvalue = np.array([hdul[i].header[header_name] for i in ext])
    else:
        hdvalue = {}
        if fix == True:
                hdul.verify('fix')
        if ext is None:
            ext_all = get_ext_all(hdul)
            [[hdvalue.update({j: np.array([hdul[i].header[j] for i in ext_all])})] for j in header_name]
        else:
            if np.size(ext) == 1:
                [hdvalue.update({j: hdul[ext].header[j]}) for j in header_name]
            else:
                [[hdvalue.update({j: np.array([hdul[i].header[j] for i in ext])})] for j in header_name]

    return hdvalue

def get_timeDt(hdul, ext=None):
    '''
    Get time (datetime) information from headers in selected extensions.
    Note that this does NOT include those of the primary (ext=0) and total (ext=1) extensions.
    arg:
        hdul: open fits file
        ext: extension (integer or list or 1d array)
    return: one datetime or a list of them
    '''
    if ext is None:
        ext_all = get_ext_all(hdul)
        timeDt = np.array([get_timeDt_mean( [ str2Dt(hdul[i].header['DATE-OBS']), str2Dt(hdul[i].header['DATE-END']) ] ) for i in ext_all])

    else:
        if np.size(ext) == 1:
            if type(ext) is int:
                sDt = str2Dt(hdul[ext].header['DATE-OBS'])
                eDt = str2Dt(hdul[ext].header['DATE-END'])
                timeDt = get_timeDt_mean([sDt,eDt])
            else:
                sDt = str2Dt(hdul[ext[0]].header['DATE-OBS'])
                eDt = str2Dt(hdul[ext[0]].header['DATE-END'])
                timeDt = get_timeDt_mean([sDt,eDt])
        else:
            timeDt = np.array([get_timeDt_mean( [str2Dt(hdul[i].header['DATE-OBS']), str2Dt(hdul[i].header['DATE-END'])] ) for i in ext])

    return timeDt

def get_ext_seorb(hdul, delta_thre=60, add_first_and_last_exts=False):

    timeDt = get_timeDt(hdul)
    ext_all = get_ext_all(hdul)
    if len(ext_all) == 0:
        return None, None
    else:
        delta = timeDt[1:] - timeDt[0:-1]
        sec_arr = np.array([idelta.seconds for idelta in delta])
        idx = np.where(sec_arr > delta_thre)[0]
        ext_e = idx + ext_offset
        ext_s = ext_e + 1

        if add_first_and_last_exts:
            ext_s = np.sort(np.append(ext_s, ext_all[0]))
            ext_e = np.append(ext_e, ext_all[-1])

        return ext_s, ext_e



def get_calflg(hdul, ext=None, string=True):
    '''
    Get calflag data from headers at the selected extensions.
    Note that this does NOT include those of the primary (ext=0) and total (ext=1) extensions.
    arg:
        hdul: open fits file
        ext: extension (integer or list or 1d array)
        string: If True, data is returned in string ("dis"/"ena") format, otherwise in float (1 or 0).
                Note that "dis"/1  and "ena"/0 correspond to planet/sky observation, respectively.
    return: Array of calflg data
    '''
    calflg = get_header_value(hdul, 'CALFLG', ext)
    calflg_v = np.array([1 if iflg == 'dis' else 0 for iflg in calflg])
    if string:
        return calflg
    else:
        return calflg_v


def get_gcbc(hdul, ext=None):
    '''
    Get guding camera's barycenter data from headers at the selected extensions.
    Note that this does NOT include those of the primary (ext=0) and total (ext=1) extensions.
    arg:
        hdul: open fits file
        ext: extension (integer or list or 1d array)
    return: Tuple of gcbc arrays (bc1x, bc1y, bc2x, bc2y)
    '''
    bc1x = get_header_value(hdul, 'BC1XAVE', ext, fix=True)
    bc1y = get_header_value(hdul, 'BC1YAVE', ext, fix=True)
    bc2x = get_header_value(hdul, 'BC2XAVE', ext, fix=True)
    bc2y = get_header_value(hdul, 'BC2YAVE', ext, fix=True)
    bc1x_2 = np.array([float(ival) for ival in bc1x])
    bc1y_2 = np.array([float(ival) for ival in bc1y])
    bc2x_2 = np.array([float(ival) for ival in bc2x])
    bc2y_2 = np.array([float(ival) for ival in bc2y])

    return bc1x_2, bc1y_2, bc2x_2, bc2y_2


def get_submod(hdul, ext=None):
    submod = get_header_value(hdul, 'SUBMOD', ext)
    return submod

def get_submst(hdul, ext=None):
    submst = get_header_value(hdul, 'SUBMST', ext)
    return submst

def get_sclt(hdul, ext=None):
    ltesc = get_header_value(hdul, 'LTESC', ext)
    return ltesc

def get_sclt_pla(hdul, ext=None):
    ltpsc = get_header_value(hdul, 'LTPSC', ext)
    return ltpsc

def get_sclon(hdul, ext=None):
    sclon = get_header_value(hdul, 'SLONESC', ext)
    return sclon

def get_sclat(hdul, ext=None):
    sclat = get_header_value(hdul, 'SLATESC', ext)
    return sclat

def get_dist_earth_sc(hdul, ext=None):
    dist_esc = get_header_value(hdul, 'RADIESC', ext)
    return dist_esc

def get_dist_earth_sun(hdul, ext=None):
    dist_esu = get_header_value(hdul, 'RADIESU', ext)
    return dist_esu

def get_dist_pla_sc(hdul, ext=None):
    dist_psc = get_header_value(hdul, 'RADIPSC', ext)
    return dist_psc

def get_dist_pla_sun(hdul, ext=None):
    dist_psu = get_header_value(hdul, 'RADIPSU', ext)
    return dist_psu

def get_slit_mode(hdul, ext=None):
    slit_mode = get_header_value(hdul, 'SLITMODE', ext)
    if isinstance(slit_mode, str):
        slit_mode = int(slit_mode.split()[0])
    else:
        slit_mode = np.array([int(imod.split()[0]) for imod in slit_mode])
    return slit_mode


def get_xslice(data, ylim, mean=False, include_err=False):
    '''
    get xslice of the 2D spectrum.
    arg:
        data: img data (1024 x 1024)
        ylim: yrange where counts to be integrated (or averaged)
    return: x (wv) profile (1024) integrated (averaged) over ylim
    '''
    if np.size(ylim) == 1:
        return data[ylim, :]
    else:
        if mean:
            if include_err:
                return np.nanmean(data[ylim[0]:ylim[1], :], axis=0), np.sqrt(np.nansum(data[ylim[0]:ylim[1], :], axis=0))/len(range(ylim[0], ylim[1]))
            else:
                return np.nanmean(data[ylim[0]:ylim[1], :], axis=0)
        else:
            if include_err:
                return np.nansum(data[ylim[0]:ylim[1], :], axis=0), np.sqrt(np.nansum(data[ylim[0]:ylim[1], :], axis=0))
            else:
                return np.nansum(data[ylim[0]:ylim[1], :], axis=0)

def get_yslice(data, xlim, mean=False, include_err=False):
    '''
    get yslice of the 2D spectrum.
    arg:
        data: img data (1024 x 1024)
        xlim: xrange where counts to be integrated (or averaged)
    return: y (spatial) profile (1024) integrated (averaged) over xlim
    '''
    if np.size(xlim) == 1:
        return data[:, xlim]
    else:
        if mean:
            if include_err:
                return np.nanmean(data[:, xlim[0]:xlim[1]], axis=1), np.sqrt(np.nansum(data[:, xlim[0]:xlim[1]], axis=1))/len(range(xlim[0], xlim[1]))
            else:
                return np.nanmean(data[:, xlim[0]:xlim[1]], axis=1)
        else:
            if include_err:
                return np.nansum(data[:, xlim[0]:xlim[1]], axis=1), np.sqrt(np.nansum(data[:, xlim[0]:xlim[1]], axis=1))
            else:
                return np.nansum(data[:, xlim[0]:xlim[1]], axis=1)

def get_total_count(data, xaxis, yaxis, xlim, ylim):
    '''
    get total count in region of interest (xlim x ylim).
    arg:
        data: img data (1024 x 1024)
        xaxis: values of x-axis (angstrom) (1024)
        yaxis: values of y-axis (arcsec)   (1024)
        xlim: xrange where counts to be integrated
        ylim: yrange where counts to be integrated
    return: total count integrated over xlim x ylim
    '''
    idx_x1 = np.abs(xaxis - xlim[0]).argmin()
    idx_x2 = np.abs(xaxis - xlim[1]).argmin()
    if idx_x1 > idx_x2:
        idx_x1, idx_x2 = idx_x2, idx_x1
    idx_y1 = np.abs(yaxis - ylim[0]).argmin()
    idx_y2 = np.abs(yaxis - ylim[1]).argmin()
    if idx_y1 > idx_y2:
        idx_y1, idx_y2 = idx_y2, idx_y1

    return np.nansum(data[idx_y1:idx_y2, idx_x1:idx_x2])


def check_y_pol(hdul, ext):
    '''Check satellite Y-axis polarlizaion'''
    SLX1DEC = get_header_value(hdul,'SLX1DEC')[ext]
    SLX3DEC = get_header_value(hdul,'SLX3DEC')[ext]
    delta_dec = SLX1DEC- SLX3DEC
    if delta_dec >= 0:
        y_pol = 1
    else:
        y_pol = 0
    return y_pol

def correct_distortion(img_ucal, y_pol, plot=False):
    '''
    Correct the distortion of 2D spectral image
    TODO: implement this function later
    Currently not used.
    """
    '''
    parent = Path(__file__).resolve().parent
    if y_pol == 1:
        x_table = sio.readsav(parent.joinpath("sav/xtable_0.sav"))
        map1 = np.array(x_table['x_table'],dtype=np.float32)
        wl = np.array(x_table['wl'],dtype=np.float32)
    else:
        x_table = sio.readsav(parent.joinpath("sav/xtable_1.sav"))
        map1 = np.array(x_table['x_table'],dtype=np.float32)
        wl = np.array(x_table['wl'],dtype=np.float32)
    y_table =  sio.readsav(parent.joinpath("sav/y_table.sav"))
    map2 = np.array(y_table['y_table'],dtype=np.float32)
    yarr = (np.arange(0,1024)- 575)*yscl  # Need to confirm what 575 indicates.

    ones = np.ones_like(img_ucal, dtype=np.float32)
    coords = np.array([map2.ravel(), map1.ravel()])
    img_cal = map_coordinates(img_ucal, coords, order=0, mode='constant').reshape(img_ucal.shape)
    img_cal = img_cal/np.sum(img_cal)*np.sum(img_ucal)
    print("Before calibration sum:", np.sum(img_ucal))
    print("After calibration sum:", np.sum(img_cal))

    if plot:
        fig = plt.figure(figsize=(12,12))
        plt.subplot(2,1,1)
        mesh_ucal = plt.pcolormesh(wl, yarr, img_ucal, cmap='inferno',norm=colors.LogNorm())
        plt.colorbar(mesh_ucal, label='counts')
        plt.xlabel('Wavelength [Å]')
        plt.ylabel('bins')
        plt.title('Uncal')
        plt.ylim(-200,200)
        plt.subplot(2,1,2)
        mesh_cal = plt.pcolormesh(wl, yarr, img_cal, cmap='inferno',norm=colors.LogNorm())
        plt.colorbar(mesh_cal, label='counts')
        plt.xlabel('Wavelength [Å]')
        plt.ylabel('bins')
        plt.title('Cal')
        plt.ylim(-200,200)

    return img_cal

########################
## plotting functions ##
########################

def plot_img(hdul, ext=None, Rayleigh=False, ax=None, **kwarg):

    if ext is None:
        ext = get_ext_all(hdul)

    img = get_img(hdul, ext)
    ndat = np.size(ext)
    timeDt = get_timeDt(hdul, ext)

    if np.size(timeDt)>1:
        Dt_mid = timeDt[int(np.size(timeDt)/2)]
    else:
        Dt_mid = timeDt
    date = str(Dt_mid.year).zfill(4) + str(Dt_mid.month).zfill(2) + str(Dt_mid.day).zfill(2)
    try:
        xcal, C2R, C2Rtbl = get_cal_daily(date)
    except FileNotFoundError:
        print('Cal file does not exist, use v0...')
        xcal, C2R, C2Rtbl = get_cal()
    ycal = np.arange(1024) ## make ycal a keyword later

    #plot
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
    #ax.set_title(target_body + '_' + date + ' ' + str(ndat) + ' min')
    if Rayleigh:
        mesh = ax.pcolormesh(xcal, ycal, img/ndat*C2Rtbl,  **kwarg)
    else:
        mesh = ax.pcolormesh(xcal, ycal, img/ndat,  **kwarg)
    return mesh

def plot_xslice(hdul, ext=None, ylim=None, ymean=False, Rayleigh=False, ax=None, **kwarg):
    '''
    test
    '''
    if ext is None:
        ext = get_ext_all(hdul)
    timeDt = get_timeDt(hdul)
    Dt_mid = timeDt[int(np.size(timeDt)/2)]
    date = str(Dt_mid.year).zfill(4) + str(Dt_mid.month).zfill(2) + str(Dt_mid.day).zfill(2)
    try:
        xcal, C2R, C2Rtbl = get_cal_daily(date)
    except FileNotFoundError:
        print('Cal file does not exist, use v0...')
        xcal, C2R, C2Rtbl = get_cal()
    ycal = np.arange(1024)
    if isinstance(ext, (int, np.integer)):
        ndat = get_nextend(hdul)
    else:
        ndat = np.size(ext)
    img = get_img(hdul, ext)
    img_err = np.sqrt(img)

    xslice, xslice_err = get_xslice(img, ylim, include_err=True)
    ny = ylim[1]-ylim[0]

    if Rayleigh:
        xprof = xslice/ndat/ny*C2R
        xprof_err = xslice_err/ndat/ny*C2R
        ylabel = 'Rayleigh/pix'
    else:
        xprof = xslice/ndat
        xprof_err = xslice_err/ndat
        ylabel = '#/min'
        if ymean:
            xprof /= np.diff(ylim)
            xprof_err /= np.diff(ylim)
            ylabel = '#/min/pix'

    #plot
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

    ax_out = ax.errorbar(xcal, xprof, xprof_err, **kwarg)
    ax.set_xlabel('wavelength [Å]')
    ax.set_ylabel(ylabel)

    return ax_out

def plot_yslice(hdul, ext=None, wvlim=None, xlim=None, xmean=False,
               Rayleigh=False, ycal=None, ycal_label='pixel', ax=None, **kwarg):

    if ext is None:
        ext = get_ext_all(hdul)
    timeDt = get_timeDt(hdul)
    Dt_mid = timeDt[int(np.size(timeDt)/2)]
    date = str(Dt_mid.year).zfill(4) + str(Dt_mid.month).zfill(2) + str(Dt_mid.day).zfill(2)
    try:
        xcal, C2R, C2Rtbl = get_cal_daily(date)
    except FileNotFoundError:
        print('Cal file does not exist, use v0...')
        xcal, C2R, C2Rtbl = get_cal()
    if ycal is None:
        ycal = np.arange(1024)
    if wvlim is not None:
        xlim = get_xbin_lim(wvlim)
    if isinstance(ext, (int, np.integer)):
        ndat = get_nextend(hdul)
    else:
        ndat = np.size(ext)
    img = get_img(hdul, ext)
    img_err = np.sqrt(img)

    yslice, yslice_err = get_yslice(img, xlim, include_err=True)
    slit_mode = get_slit_mode(hdul, 2)
    nx_slit = slit_mode/10

    if Rayleigh:
        C2Ravg = np.mean(C2R[xlim[0]:xlim[1]])
        yprof = yslice/ndat/nx_slit*C2Ravg
        yprof_err = yslice_err/ndat/nx_slit*C2Ravg
        ylabel = 'Rayleigh/pix'
    else:
        yprof = yslice/ndat
        yprof_err = yslice_err/ndat
        ylabel = '#/min'
        if xmean:
            yprof /= nx_slit
            yprof_err /= nx_slit
            ylabel = '#/min/pix'

    #plot
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

    ax_out = ax.errorbar(ycal, yprof, yprof_err, **kwarg)
    ax.set_xlabel(ycal_label)
    ax.set_ylabel(ylabel)
    return ax_out



def plot_xprof(hdul, ext=None, ylim=None, wvlim=None, avgpixel=False, Rayleigh=False, ax=None, **kwarg):
    '''
    This function is replaced with plot_xlsice() and will be obsoleted.
    '''
    if ext is None:
        ext = get_ext_all(hdul)
    timeDt = get_timeDt(hdul)
    Dt_mid = timeDt[int(np.size(timeDt)/2)]
    date = str(Dt_mid.year).zfill(4) + str(Dt_mid.month).zfill(2) + str(Dt_mid.day).zfill(2)
    try:
        xcal, C2R, C2Rtbl = get_cal_daily(date)
    except FileNotFoundError:
        print('Cal file does not exist, use v0...')
        xcal, C2R, C2Rtbl = get_cal()
    ycal = np.arange(1024)
    if isinstance(ext, (int, np.integer)):
        ndat = get_nextend(hdul)
    else:
        ndat = np.size(ext)
    img = get_img(hdul, ext)
    img_err = np.sqrt(img)
    if Rayleigh:
        xprof = np.nansum(img[ylim[0]:ylim[1], :], axis=0)/ndat*C2R
        xprof_err = np.sqrt(np.nansum(img_err[ylim[0]:ylim[1], :]**2, axis=0))/ndat*C2R
    else:
        xprof = np.nansum(img[ylim[0]:ylim[1], :], axis=0)/ndat
        xprof_err = np.sqrt(np.nansum(img_err[ylim[0]:ylim[1], :]**2, axis=0))/ndat

    if avgpixel:
        xprof /= np.diff(ylim)
        xprof_err /= np.diff(ylim)

    #plot
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

    if wvlim is not None:
        xlim = get_xbin_lim(wvlim)
        ymax = np.max(xprof[xlim[0]:xlim[1]])
        ax.set_ylim(top=ymax*1.1)

    ax_out = ax.errorbar(xcal, xprof, xprof_err, **kwarg)

    return ax_out

def plot_yprof(hdul, ext=None, xlim=None, wvlim=None, ycal=None, avgpixel=False, Rayleigh=False, ax=None, **kwarg):
    '''
    This function is replaced with plot_ylsice() and will be obsoleted.
    '''
    if ext is None:
        ext = get_ext_all(hdul)
    timeDt = get_timeDt(hdul)
    Dt_mid = timeDt[int(np.size(timeDt)/2)]
    date = str(Dt_mid.year).zfill(4) + str(Dt_mid.month).zfill(2) + str(Dt_mid.day).zfill(2)
    try:
        xcal, C2R, C2Rtbl = get_cal_daily(date)
    except FileNotFoundError:
        print('Cal file does not exist, use v0...')
        xcal, C2R, C2Rtbl = get_cal()
    if ycal is None:
        ycal = np.arange(1024)
    if wvlim is not None:
        xlim = get_xbin_lim(wvlim)
    if isinstance(ext, (int, np.integer)):
        ndat = get_nextend(hdul)
    else:
        ndat = np.size(ext)
    img = get_img(hdul, ext)
    img_err = np.sqrt(img)
    if Rayleigh:
        C2Ravg = np.mean(C2R[xlim[0]:xlim[1]])
        yprof = np.nansum(img[:, xlim[0]:xlim[1]], axis=1)/ndat*C2Ravg
        yprof_err = np.sqrt(np.nansum(img_err[:, xlim[0]:xlim[1]]**2, axis=1))/ndat*C2Ravg
    else:
        yprof = np.nansum(img[:, xlim[0]:xlim[1]], axis=1)/ndat
        yprof_err = np.sqrt(np.nansum(img_err[:, xlim[0]:xlim[1]]**2, axis=1))/ndat

    if avgpixel:
        yprof /= np.diff(xlim)
        yprof_err /= np.diff(xlim)

    #plot
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
    ax_out = ax.errorbar(ycal, yprof, yprof_err, **kwarg)
    return ax_out


def download_data_l2_pub(target, date, mode='*', lv='02', vr='00'):
    '''
    Download the Hisaki Level-2 data from a public data server at ISAS/DARTS
       args:
        target:     target name. e.g. 'jupiter'
        date:       date. e.g. '20240101'
        mode:       observation mode
        lv:         Level number
        vr:         File version
       returns:
        N.A.
    '''
    # Source
    fn = 'exeuv.' + target + '.mod.' + mode + '.' + date + '.lv.' + lv + '.vr.'+ vr + '.fits'
    url = url_l2_pub + fn

    # Destination
    dir = dataloc
    os.makedirs(dir, exist_ok=True)
    fn_full = os.path.join(dir, fn)
    is_file = os.path.isfile(fn_full)

    if is_file:
        print("File "+fn+" exists in the local computer.")
    else:
        response = requests.get(url)
        # Check status
        if response.status_code == 200:
            print("File "+fn+" is downloading to the local computer.")
            with open(fn_full, 'wb') as f:
                f.write(response.content)
        else:
            print('Download failed:', response.status_code, response.reason)

def download_data_l2(target, date, mode='*', lv='02', vr='00', uname='hisaki', passwd=None):
    '''
    Download the Hisaki Level-2 data from an access limitted data server at ISAS/DARTS
       args:
        target:     target name. e.g. 'jupiter'
        date:       date. e.g. '20240101'
        mode:       observation mode
        lv:         Level number
        vr:         File version
        uname:      user name
        passwd:     password 
       returns:
        N.A.
    '''
    # Source
    fn = 'exeuv.' + target + '.mod.' + mode + '.' + date + '.lv.' + lv + '.vr.'+ vr + '.fits'
    url = url_l2 + fn

    # Destination
    dir = dataloc
    os.makedirs(dir, exist_ok=True)
    fn_full = os.path.join(dir, fn)
    is_file = os.path.isfile(fn_full)

    if is_file:
        print("File "+fn+" exists in the local computer.")
    else:
        response = requests.get(url, auth=(uname, passwd))
        # Check status
        if response.status_code == 200:
            print("File "+fn+" is downloading to the local computer.")
            with open(fn_full, 'wb') as f:
                f.write(response.content)
        else:
            print('Download failed:', response.status_code, response.reason)

#############
## L2prime ##
#############

def download_data_l2p(target, date, lv='02p', lt='00-24', dt='00106', vr='01_00', sky=False):
    '''
    Download the Hisaki Level-2 prime data from a data server at Tohoku Univ.
       args:
        target:     target name. e.g. 'jupiter'
        date:       date. e.g. '20240101'
        lv:         Level number
        lt:         Local time range of the data '00-24' or '20-04'
        dt:         Time step '00010' or '00106'
        vr:         File version
        sky:        If Ture, download sky data
       returns:
        fn:         Full path to the downloaded data file
    '''
    yr = date[0:4]
    if sky==False:
        fn = 'exeuv_'+ target + '_' + date + '_lv' + lv + '_LT' + lt + '_dt' + dt + '_vr'+ vr + '.fits'
        url = url_l2p + target + '/' + target + '/' + yr + '/' + fn
        dir = os.path.join(dataloc_l2p, target, target, yr)
    else:
        fn = 'exeuv_sky_' + date + '_lv' + lv + '_LT' + lt + '_dt' + dt + '_vr'+ vr + '.fits'
        url = url_l2p + target + '/sky/' + yr + '/' + fn
        dir = os.path.join(dataloc_l2p, target, 'sky', yr)
    os.makedirs(dir, exist_ok=True)
    fn_full = os.path.join(dir, fn)
    is_file = os.path.isfile(fn_full)
    if is_file:
        print("File "+fn+" exists in the local computer.")
    else:
        response = requests.get(url)
        # Check status
        if response.status_code == 200:
            print("File "+fn+" is downloading to the local computer.")
            with open(fn_full, 'wb') as f:
                f.write(response.content)
        else:
            print('Download failed:', response.status_code, response.reason)

    return(fn_full)

def download_data_l3(target, date, lv='03', lt='00-24', dt='00010', vr='01_00'):
    '''
    Download the Hisaki Level-3 data from a data server at Tohoku Univ.
       args:
        target:     target name. e.g. 'aurora' or 'torus'
        date:       date. e.g. '20240101'
        lv:         Level number
        lt:         Local time range of the data '00-24'
        dt:         Time step '00010' or '00106'
        vr:         File version
       returns:
        fn:         Full path to the downloaded data file
    '''
    if target == 'aurora':
        target_dir = target
    else:
        target_dir = 'roi50-70'
    fn = 'exeuv_'+ target + '_' + date + '_lv' + lv + '_LT' + lt + '_dt' + dt + '_vr'+ vr + '.fits'
    url = url_l3 + target_dir + '/' + fn
    dir = os.path.join(dataloc_l3, target_dir)

    os.makedirs(dir, exist_ok=True)
    fn_full = os.path.join(dir, fn)
    is_file = os.path.isfile(fn_full)
    if is_file:
        print("File "+fn+" exists in the local computer.")
    else:
        response = requests.get(url)
        # Check status
        if response.status_code == 200:
            print("File "+fn+" is downloading to the local computer.")
            with open(fn_full, 'wb') as f:
                f.write(response.content)
        else:
            print('Download failed:', response.status_code, response.reason)

    return(fn_full)

def get_xaxis(hdul, ext=1):
    NAXIS1 = int(hdul[ext].header['NAXIS1'])
    CRVAL1 = float(hdul[ext].header['CRVAL1'])
    CRPIX1 = float(hdul[ext].header['CRPIX1'])
    CDELT1 = float(hdul[ext].header['CDELT1'])
    x_axis = CRVAL1 + (np.arange(NAXIS1) - CRPIX1) * CDELT1
    return x_axis

def get_yaxis(hdul, ext=1):
    NAXIS2 = int(hdul[ext].header['NAXIS2'])
    CRVAL2 = float(hdul[ext].header['CRVAL2'])
    CDELT2 = float(hdul[ext].header['CDELT2'])
    CRPIX2 = float(hdul[ext].header['CRPIX2'])
    y_axis = CRVAL2 + (np.arange(NAXIS2) - CRPIX2) * CDELT2
    return y_axis

def get_labels(hdul, ext=1):
    x_label = hdul[ext].header['CUNIT1']
    y_label = hdul[ext].header['CUNIT2']
    BUNITS = hdul[ext].header['BUNITS']
    return x_label, y_label, BUNITS

def check_y_pol(hskdat):
    '''
    Check satellite Y-axis polarlizaion
       args:
        hskdat
       returns:
        y_pol:      Y-axis polarlizaion
    '''
    SLX1DEC = hskdat.get_header_value('SLX1DEC')[0]
    SLX3DEC = hskdat.get_header_value('SLX3DEC')[0]
    delta_dec = SLX1DEC- SLX3DEC
    if delta_dec >= 0:
        y_pol = 1
    else:
        y_pol = 0
    return y_pol


def exc_cal_venus(hskdat,n_img =1):
    y_pol = check_y_pol(hskdat)
    if y_pol == 1:
        x_table = sio.readsav("xtable_0.sav")
        map1 = np.array(x_table['x_table'],dtype=np.float32)
        wl = np.array(x_table['wl'],dtype=np.float32)
    else:
        x_table = sio.readsav("xtable_1.sav")
        map1 = np.array(x_table['x_table'],dtype=np.float32)
        wl = np.array(x_table['wl'],dtype=np.float32)
    y_table =  sio.readsav("y_table.sav")
    map2 = np.array(y_table['y_table'],dtype=np.float32)
    pscl_s  = 4.23 #exceed plate scale (y-axis) [arcsec/pix]
    yarr = (np.arange(0,1024)- 575)*pscl_s

    
    img_ucal = hskdat.get_img(n_img)
    ones = np.ones_like(img_ucal, dtype=np.float32)
    coords = np.array([map2.ravel(), map1.ravel()])
    img_cal = map_coordinates(img_ucal, coords, order=0, mode='constant').reshape(img_ucal.shape)
    img_cal = img_cal/np.sum(img_cal)*np.sum(img_ucal)
    print("Before calibration sum:", np.sum(img_ucal))
    print("After calibration sum:", np.sum(img_cal))

 
    fig = plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    mesh_ucal = plt.pcolormesh(wl, yarr, img_ucal, cmap='inferno',norm=colors.LogNorm())
    plt.colorbar(mesh_ucal, label='counts')
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('bins')
    plt.title('Uncal')
    plt.ylim(-200,200)
    plt.subplot(2,1,2)
    mesh_cal = plt.pcolormesh(wl, yarr, img_cal, cmap='inferno',norm=colors.LogNorm())
    plt.colorbar(mesh_cal, label='counts')
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('bins')
    plt.title('Cal')
    plt.ylim(-200,200)

    return img_cal