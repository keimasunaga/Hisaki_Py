import glob
import numpy as np
import astropy.io.fits as fits
from datetime import timedelta
import matplotlib.pyplot as plt

from .env import get_env
from .time import str2Dt
from .calib import get_cal_daily, get_cal, get_xbin_lim

# Hisaki data location
dataloc = get_env('dataloc_hsk')

# Characteristic extensions
ext_primary = 0 # Primary
ext_total = 1   # Total
ext_offset = 2  # Offset

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

    def get_nextend(self):
        self.nextend = self.hdul[ext_primary].header['NEXTEND']
        return self.nextend

    def get_ext_all(self):
        self.get_nextend()
        return list(range(self.ext_offset, self.nextend))

    def get_img(self, ext=1):
        return get_img(self.hdul, ext)

    def get_header(self, ext):
        return self.hdul[ext].header

    def get_header_value(self, header_name, ext=None):
        return get_header_value(self.hdul, header_name, ext)

    def get_timeDt(self, ext=None):
        return get_timeDt(self.hdul, ext)

    def get_cal(self, daily=False):
        if daily:
            self.xcal_daily, self.C2R_daily, self.C2Rtbl_daily = get_cal_daily(self.date)
            return self.xcal_daily, self.C2R_daily, self.C2Rtbl_daily
        else:
            self.xcal, self.C2R, self.C2Rtbl = get_cal()
            return self.xcal, self.C2R, self.C2Rtbl

    def get_ext_seorb(self, delta_thre=2500):
        return get_ext_seorb(self.hdul, delta_thre)



def get_fname(target, date='*', mode='*', lv='02', vr='00', fullpath=False):
    pattern = 'exeuv.'+ target + '.mod.' + mode + '.' + date + '.lv.' + lv + '.vr.' + vr + '.fits'
    filepath = glob.glob(dataloc + '/' + pattern)
    if fullpath:
        if np.size(filepath) == 1:
            filepath = filepath[0]
        if np.size(fullpath) == 0:
            print('---- No data found, returning an empty list ----')
        return filepath
    else:
        if np.size(filepath) == 1:
            fname = filepath[0].split('/')[-1]
        else:
            fname = [ifname.split('/')[-1] for ifname in filepath]
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
    fname = path[0].split('/')[-1]
    print('---- Opened ' + fname + ' ----')
    print('---- Make sure that the fits data should be closed eventually. ----')
    return hdul

def fitsclose(hdul):
    hdul.close()
    print('---- The fits data closed. ----')

def get_nextend(hdul):
    nextend = hdul[ext_primary].header['NEXTEND']
    return nextend

def get_ext_all(hdul):
    nextend = get_nextend(hdul)
    return list(range(ext_offset, nextend))

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
            n_ext = hdul[0].header['NEXTEND']
            hdvalue = np.array([hdul[i].header[header_name] for i in range(ext_offset, n_ext)])
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
            n_ext = hdul[0].header['NEXTEND']
            [[hdvalue.update({j: np.array([hdul[i].header[j] for i in range(ext_offset, n_ext)])})] for j in header_name]
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
        n_ext = hdul[0].header['NEXTEND']
        timeDt = np.array([str2Dt(hdul[i].header['DATE-OBS']) + timedelta(seconds=30) for i in range(ext_offset, n_ext)])

    else:
        if np.size(ext) == 1:
            if type(ext) is int:
                timeDt = str2Dt(hdul[ext].header['DATE-OBS']) + timedelta(seconds=30)
            else:
                timeDt = str2Dt(hdul[ext[0]].header['DATE-OBS']) + timedelta(seconds=30)
        else:
            timeDt = np.array([str2Dt(hdul[i].header['DATE-OBS']) + timedelta(seconds=30) for i in ext])

    return timeDt

def get_ext_seorb(hdul, delta_thre=2500):

    timeDt = get_timeDt(hdul)
    ext_offset = 2
    n_ext = np.size(timeDt)
    ext_all = list(range(ext_offset, ext_offset+n_ext))
    if len(ext_all) == 0:
        return None, None
    else:
        delta = timeDt[1:-1] - timeDt[0:-2]
        sec_arr = np.array([idelta.seconds for idelta in delta])
        idx = np.where(sec_arr > delta_thre)[0]
        ext_e = idx + ext_offset
        ext_s = ext_e + 1
        ext_s2 = np.append(ext_all[0], ext_s)
        ext_e2 = np.append(ext_e, ext_all[-1])

        return ext_s2, ext_e2



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



########################
## plotting functions ##
########################

def plot_img(hdul, ext=None, Rayleigh=False, ax=None, **kwarg):

    if ext is None:
        ext = get_ext_all(hdul)
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
    if isinstance(ext, (int, np.integer)):
        ndat = get_nextend(hdul)
    else:
        ndat = np.size(ext)
    img = get_img(hdul, ext)

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

def plot_xprof(hdul, ext=None, ylim=None, wvlim=None, avgpixel=False, Rayleigh=False, ax=None, **kwarg):

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
