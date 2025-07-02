import numpy as np
import os
from pathlib import Path
import astropy.io.fits as fits
from .env import get_env

# Hisaki calibration data location
calloc = get_env('hsk_cal_data_loc')

def get_cal():
    '''
    Get information of an early calibration data.
    The calibration data generally should not be used to all data.
    Use it only when daily calibration data is not published yet.
    Unit of C2Rtbl: Rayleigh/(cnts/min); (cnts/min)/pixel to Rayleigh/pixel
    '''
    parent = Path(__file__).resolve().parent
    path = parent.joinpath('npy/caldata.npy')
    dic = np.load(path, allow_pickle=True).item()
    xcal = dic['xcal']
    C2R = dic['C2R']
    C2Rtbl = dic['C2Rtbl']
    return xcal, C2R, C2Rtbl


def get_cal_daily(date):
    '''
    Get information of daily calibration data.
    arg:
        date: as a format of yyyymmdd
    return:
        tuple (xcal, C2R, C2Rtbl)
    Unit of C2Rtbl: Rayleigh/(cnts/min); (cnts/min)/pixel to Rayleigh/pixel
    '''

    fname_cal = 'calib_' + date + '_v1.0.fits'
    path_cal = calloc + fname_cal
    with fits.open(path_cal) as hdul_cal:
        xcal = hdul_cal[1].data[550]  ## 550 is just a random y bin because the cal data has no dependence in the y direction
        C2Rtbl = hdul_cal[3].data
        C2R = hdul_cal[3].data[550]
    return xcal, C2R, C2Rtbl


def get_cal_ver(lv='l2', vr='1.0'):
    '''
    Get information of calibration data.
    arg:
        lv: 'l2' or 'l2p'
        vr: '1.0' or '1.1'
    return:
        tuple (xcal, ycal, C2R, C2Rtbl) for l2
        tuple (xcal, ycal, C2Rtbl_n, C2Rtbl_s) for l2
            C2Rtbl_n is table for northward Y-POL
            C2Rtbl_s is table for southward Y-POL
    Unit of C2Rtbl: Rayleigh/(cnts/min); (cnts/min)/pixel to Rayleigh/pixel
   '''

    fname_cal = 'calib_' + lv + '_v' + vr + '.fits'
    path_cal = calloc + fname_cal
    with fits.open(path_cal) as hdul_cal:
        if lv=='l2p':
            NAXIS1 = int(hdul_cal[1].header['NAXIS1'])
            CRVAL1 = float(hdul_cal[1].header['CRVAL1'])
            CRPIX1 = float(hdul_cal[1].header['CRPIX1'])
            CDELT1 = float(hdul_cal[1].header['CDELT1'])
            xcal = CRVAL1 + (np.arange(NAXIS1) - CRPIX1) * CDELT1
            NAXIS2 = int(hdul_cal[1].header['NAXIS2'])
            CRVAL2 = float(hdul_cal[1].header['CRVAL2'])
            CDELT2 = float(hdul_cal[1].header['CDELT2'])
            CRPIX2 = float(hdul_cal[1].header['CRPIX2'])
            ycal = CRVAL2 + (np.arange(NAXIS2) - CRPIX2) * CDELT2
            C2Rtbl_n = hdul_cal[1].data
            C2Rtbl_s = hdul_cal[2].data
            return xcal, ycal, C2Rtbl_n, C2Rtbl_s
        else:
            xcal = hdul_cal[1].data[550]   ## 550 is just a random y bin because the cal data has no dependence in the y direction
            ycal = hdul_cal[2].data[:,550]
            C2Rtbl = hdul_cal[3].data
            C2R = hdul_cal[3].data[550]
            return xcal, ycal, C2R, C2Rtbl


def get_nearest_xbin(wv, shift=0):
    """
    returns the nerest index of xcal from the input wavelength
    arg:
        wv: wavelength in angstrom
        shift: shifting wavelength in angstrom
    return:
        the nerest index of xcal data
    """
    xcal, _, _ = get_cal()
    diff = xcal - (wv + shift)
    idx = np.abs(diff).argmin()
    return idx

def get_xbin_lim(wv_lim):
    idx0 = get_nearest_xbin(wv_lim[0])
    idx1 = get_nearest_xbin(wv_lim[1])
    xbin_lim = [idx0, idx1]
    xbin_lim.sort()
    return xbin_lim


class CalibData:
    def __init__(self, date):
        self.date = date
        try:
            self.xcal, self.C2R, self.C2Rtbl = get_cal_daily(date)
        except FileNotFoundError:
            print('Cal file does not exist, use v0...')
            self.xcal, self.C2R, self.C2Rtbl = get_cal()
        self.ycal = np.arange(1024)

    def get_C2R_mean(self, xlim):
        return np.mean(self.C2R[xlim[0]:xlim[1]])

    def get_ycal_asec(self, ycent=570):
        return (self.ycal - ycent)*4.2

    def get_ycal_rpla(self, appdia, ycent=570):
        rpla_asec = appdia/2
        rpla = (self.ycal - ycent)*4.2/rpla_asec
        return rpla
