import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import astropy.io.fits as fits

import common.tools as ctools
from .data import HskData, get_fname, fitsopen, get_nextend, get_ext_all, get_img, get_ext_seorb
from .variables import dataloc, ext_primary, ext_total, ext_offset, xscl, yscl
from .calib import get_cal, get_cal_daily
from common.tools import shift_grids, AnchoredHScaleBar


def asec2rad(x_asec):
    return x_asec * np.pi/(180*3600)

def get_omega(appdia):
    return np.pi/4*(np.pi*appdia/3600/180)**2

def get_img_mlt(filenames, exts):
    '''
    returns an accumulated image for multiple days based on selected extensions (2D spectrum)
    arg:
        filenames: a list of filenames of Hisaki data.
                  ex. exeuv.mars.mod.01.yyyymmdd.lv.02.vr.00.fits
        exts: a list of lists of extensions
    return: An image data (np.array([1024, 1024]))
    '''
    data_mlt = np.zeros([1024, 1024])
    for filename, ext in zip(filenames, exts):
        data = get_img(filename, ext)
        data_mlt += data
    return data_mlt


def get_adjust_factor(img1, nimg1, img2, nimg2, xlim, ylim):
    c_on = np.sum(img1[ylim[0]:ylim[1], xlim[0]:xlim[1]])/nimg1
    c_on_err = np.sqrt(c_on)/nimg1
    c_bg = np.sum(img2[ylim[0]:ylim[1], xlim[0]:xlim[1]])/nimg2
    c_bg_err = np.sqrt(c_bg)/nimg2
    f = c_on/c_bg
    f_err = np.sqrt((c_on_err/c_bg)**2 + (c_on*c_bg_err/c_bg**2)**2)
    return f, f_err


def get_counts_roi(data, xlim, ylim, mean=False):
    '''
    get counts within the roi
    '''
    if mean:
        return np.nanmean(data[ylim[0]:ylim[1], xlim[0]:xlim[1]])
    else:
        return np.nansum(data[ylim[0]:ylim[1], xlim[0]:xlim[1]])

def get_nroi(xlim, ylim):
    '''
    get number of pixels within the roi
    '''
    return len(range(xlim[0], xlim[1])) * len(range(ylim[0], ylim[1]))

def get_xslice(data, ylim, mean=False, include_err=False):
    '''
    get xslice of the 2D spectrum.
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


def get_summary(img, ndat, img_sky, ndat_sky, xlim, ylim, ylim_adj=None):

    if ylim_adj is not None:
        f_adj, f_err =  get_adjust_factor(img, ndat, img_sky, ndat_sky, xlim, ylim_adj)

    else:
        f_adj = 1
        f_err = 0

    img_err = np.sqrt(img)
    img_sky_err = np.sqrt(img_sky)

    img_mean = img/ndat
    img_mean_err = img_err/ndat
    img_sky_mean = img_sky*f_adj/ndat_sky
    img_sky_mean_err = np.sqrt((img_sky_err*f_adj)**2 + (img_sky*f_err)**2)/ndat_sky#img_sky_err/ndat_sky*f_adj

    img_sub = img_mean - img_sky_mean
    img_sub_err = np.sqrt(img_mean_err**2 + img_sky_mean_err**2)

    xslice, xslice_err = get_xslice(img, ylim, include_err=True)
    yslice, yslice_err = get_yslice(img, xlim, include_err=True)
    xslice_mean = xslice/ndat
    yslice_mean = yslice/ndat
    xslice_mean_err = xslice_err/ndat
    yslice_mean_err = yslice_err/ndat

    xslice_sky, xslice_sky_err = get_xslice(img_sky, ylim, include_err=True)
    yslice_sky, yslice_sky_err = get_yslice(img_sky, xlim, include_err=True)
    xslice_sky_mean = xslice_sky*f_adj/ndat_sky
    yslice_sky_mean = yslice_sky*f_adj/ndat_sky
    xslice_sky_mean_err = np.sqrt((xslice_sky_err*f_adj)**2 + (xslice_sky*f_err)**2)/ndat_sky
    yslice_sky_mean_err = np.sqrt((yslice_sky_err*f_adj)**2 + (yslice_sky*f_err)**2)/ndat_sky

    xslice_sub = xslice_mean - xslice_sky_mean
    yslice_sub = yslice_mean - yslice_sky_mean
    xslice_sub_err = np.sqrt(xslice_mean_err**2 + xslice_sky_mean_err**2)
    yslice_sub_err = np.sqrt(yslice_mean_err**2 + yslice_sky_mean_err**2)

    dic = {'img_mean':img_mean, 'img_sky_mean':img_sky_mean, 'img_sub':img_sub,
           'xslice_mean':xslice_mean, 'xslice_sky_mean':xslice_sky_mean, 'xslice_sub':xslice_sub,
           'yslice_mean':yslice_mean, 'yslice_sky_mean':yslice_sky_mean, 'yslice_sub':yslice_sub,
           'img_mean_err':img_mean_err, 'img_sky_mean_err':img_sky_mean_err, 'img_sub_err':img_sub_err,
           'xslice_mean_err':xslice_mean_err, 'xslice_sky_mean_err':xslice_sky_mean_err, 'xslice_sub_err':xslice_sub_err,
           'yslice_mean_err':yslice_mean_err, 'yslice_sky_mean_err':yslice_sky_mean_err, 'yslice_sub_err':yslice_sub_err,
           'f_adj':f_adj}

    return dic



class Img():
    '''
    Image class to handle a Hisaki 2D spectrum.
    '''
    def __init__(self, counts=None, err=None, ndat=None, counts_type='total', vunit='#/pix',
                 xcal=np.arange(1024), xcal_type='bin', xunit='bin number',
                 ycal=np.arange(1024), ycal_type='bin', yunit='bin number',
                 C2R=None, C2Rtbl=None, adjusted=False, timeDt=None):
        self.counts = counts
        self.err = err
        self.ndat = ndat
        self.counts_type = counts_type
        self.xcal = xcal
        self.xcal_type = xcal_type
        self.xunit = xunit
        self.ycal = ycal
        self.ycal_type = ycal_type
        self.yunit = yunit
        self.C2R = C2R
        self.C2Rtbl = C2Rtbl
        self.vunit = vunit
        self.adjusted = adjusted
        self.timeDt = timeDt

    ## Object generator ##
    @classmethod
    def genobj(cls, counts, ndat):
        err = np.sqrt(counts)
        return cls(counts, err, ndat)

    def add_timeDt(self, timeDt):
        self.timeDt = timeDt

    def get_timeDt_mean(self):
        if self.timeDt is not None:
            return ctools.get_timeDt_mean(timeDt)
        else:
            return None

    ## Data processors ##
    def acc_data(self, other):
        if self.counts_type == 'total':
            counts_acc = self.counts + other.counts
            err_acc = np.sqrt((self.err)**2 + (other.err)**2)
            ndat_acc = self.ndat + other.ndat
            if self.timeDt is not None and other.timeDt is not None:
                timeDt_arr = np.append(self.timeDt, other.timeDt)
            else:
                timeDt_arr = None
            return Img(counts_acc, err_acc, ndat_acc, 'total', '#/pix',
                       self.xcal, self.xcal_type, self.xunit, self.ycal, self.ycal_type, self.yunit, self.C2R, self.C2Rtbl, self.adjusted, timeDt_arr)
        else:
            print('---- counts_type is not "total" ----')

    def mean(self):
        if self.counts_type == 'mean':
            raise ValueError('---- counts type is already mean ----')
        else:
            if self.timeDt is not None:
                timeDt_mean = ctools.get_timeDt_mean(self.timeDt)
            else:
                timeDt_mean = None
            return Img(self.counts/self.ndat, self.err/self.ndat, self.ndat, 'mean', '#/min/pix',
                       self.xcal, self.xcal_type, self.xunit, self.ycal, self.ycal_type, self.yunit, self.C2R, self.C2Rtbl, self.adjusted, timeDt_mean)

    def sub_data(self, other):
        if self.counts_type == 'mean' and other.counts_type == 'mean':
            counts_sub = self.counts - other.counts
            err_sub = np.sqrt(self.err**2 + other.err**2)
            return Img(counts_sub, err_sub, 1, 'sub', '#/min/pix',
                       self.xcal, self.xcal_type, self.xunit, self.ycal, self.ycal_type, self.yunit, self.C2R, self.C2Rtbl, self.adjusted, self.timeDt)

    def adjust_bg(self, img, xlim, ylim):

        c_sky = np.sum(self.counts[ylim[0]:ylim[1], xlim[0]:xlim[1]])
        c_sky_err = np.sqrt(np.sum((self.counts[ylim[0]:ylim[1], xlim[0]:xlim[1]])**2))
        c_on = np.sum(img.counts[ylim[0]:ylim[1], xlim[0]:xlim[1]])
        c_on_err = np.sqrt(np.sum((img.counts[ylim[0]:ylim[1], xlim[0]:xlim[1]])**2))
        f = c_on/c_sky
        f_err = np.sqrt((c_on_err/c_sky)**2 + (c_on*c_sky_err/c_sky**2)**2)
        counts_new = self.counts * f
        err_new = np.sqrt((f*self.err)**2 + (self.counts*f_err)**2)
        if self.counts_type == 'total' and img.counts_type ==  'total':
            return Img(counts_new, err_new, img.ndat, self.counts_type, self.vunit,
                       self.xcal, self.xcal_type, self.xunit, self.ycal, self.ycal_type, self.yunit, self.C2R, self.C2Rtbl, True, self.timeDt)
        elif self.counts_type == 'mean' and img.counts_type ==  'mean':
            return Img(counts_new, err_new, self.ndat, self.counts_type, self.vunit,
                       self.xcal, self.xcal_type, self.xunit, self.ycal, self.ycal_type, self.yunit, self.C2R, self.C2Rtbl, True, self.timeDt)
        else:
            raise ValueError('---- both counts_type should be ["total", "total"] or ["mean", "mean"] ----')

    def flip_n_roll(self, ycnt):
        """
        Flip the img and shift all data along the yaxis.
        arg:
            ycnt: the data is fliped around this y position.
        """
        idx = np.arange(np.size(self.counts[0]))
        idx_flip = np.flip(idx)
        counts_flip = np.flip(self.counts, axis=0)
        ycnt2 = np.where(idx_flip == ycnt)[0]
        shift = -(ycnt2 - ycnt)
        counts_roll = np.roll(counts_flip, shift, axis=0)

        err_flip = np.flip(self.err, axis=0)
        err_roll = np.roll(err_flip, shift, axis=0)

        return Img(counts_roll, err_roll, self.ndat, self.counts_type, self.vunit,
                   self.xcal, self.xcal_type, self.xunit, self.ycal, self.ycal_type, self.yunit, self.C2R, self.C2Rtbl, True, self.timeDt)

    #### Get counts in the 2D spectrum ####
    def get_counts_roi(self, roi_x, roi_y, mean=False):

        if self.counts_type == 'total' or 'mean':
            if mean:
                return get_counts_roi(self.counts, roi_x, roi_y, mean=True)

            else:
                return get_counts_roi(self.counts, roi_x, roi_y, mean=False)
        else:
            print('---- counts_type is not "total" or "mean" ----')

    def get_counts_roi_err(self, roi_x, roi_y, mean=False):
        nroi = hsktools.get_nroi(roi_x, roi_y)
        if self.counts_type == 'total':
            if mean:
                return np.sqrt(self.get_counts_roi(roi_x, roi_y))/nroi
            else:
                return np.sqrt(self.get_counts_roi(roi_x, roi_y))

        elif self.counts_type == 'err':
            if mean:
                return np.sqrt(np.nansum(self.counts[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]**2))/nroi
            else:
                return np.sqrt(np.nansum(self.counts[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]**2))

    def get_xslice(self, ylim, mean=False):
        '''
        Return xslice of the 2D spectrum.
        '''
        if np.size(ylim) == 1:
            self.xslice = self.counts[ylim, :]
            return self.xslice
        else:
            if mean:
                return np.nanmean(self.counts[ylim[0]:ylim[1], :], axis=0)
            else:
                return np.nansum(self.counts[ylim[0]:ylim[1], :], axis=0)

    def get_yslice(self, xlim, mean=False):
        '''
        Return yslice of the 2D spectrum.
        '''
        if np.size(xlim) == 1:
            return self.counts[:, xlim]
        else:
            if mean:
                return np.nanmean(self.counts[:, xlim[0]:xlim[1]], axis=1)
            else:
                return np.nansum(self.counts[:, xlim[0]:xlim[1]], axis=1)

    #### Handling cal data ####
    def add_calib_data(self, date=None):
        """Add xcal, C2R, and C2R table """
        if date is None:
            self.xcal, self.C2R, self.C2Rtbl = get_cal()
        else:
            self.xcal, self.C2R, self.C2Rtbl = get_cal_daily(date)
        self.xcal_type = 'wavelength'
        self.xunit = 'Å'

    def add_ycal(self, ycal, ycal_type, yunit):
        """
        Add ycal, ycal_type, and yunit
        such as r_pla, 'distance', and 'Rm'
        """
        self.ycal = ycal
        self.ycal_type = ycal_type
        self.yunit = yunit

    def ang2nm(self):
        if self.xcal_type == 'wavelength' and self.xunit == 'Å':
            self.xcal = self.xcal * 0.1
            self.xunit = 'nm'

    def nm2ang(self):
        if self.xcal_type == 'wavelength' and self.xunit == 'nm':
            self.xcal = self.xcal * 10
            self.xunit = 'Å'

    def cnts2ray(self, date=None):
        if self.ndat == 1:
            self.rayleigh = self.counts * self.C2Rtbl
            self.rayleigh_err = self.err * self.C2Rtbl
        else:
            print('---- ndat is not 1, get mean() or err_mean() ----')

    #### Visualization methods ####
    def plot(self, rayleigh=False, *args, **kwargs):
        '''
        Plots a 2D spectrum
        '''
        shifted_x = np.r_[self.xcal[0], self.xcal[1:] - np.diff(self.xcal)*0.5, self.xcal[-1]]
        shifted_y = np.r_[self.ycal[0], self.ycal[1:] - np.diff(self.ycal)*0.5, self.ycal[-1]]

        if rayleigh:
            mesh = plt.pcolormesh(shifted_x, shifted_y, self.rayleigh, *args, **kwargs)
            cbar = plt.colorbar(mesh)
            cbar.set_label('Rayleigh/pix', rotation=270, labelpad=10)
        else:
            mesh = plt.pcolormesh(shifted_x, shifted_y, self.counts, *args, **kwargs)
            cbar = plt.colorbar(mesh)
            cbar.set_label(self.vunit, rotation=270, labelpad=10)

        if self.xcal_type == 'bin':
            plt.xlabel('spectral bin')

        else:
            plt.xlabel(self.xcal_type + ' [' + self.xunit + ']')
        if self.ycal_type == 'bin':
            plt.ylabel('spatial bin')
        else:
            plt.ylabel(self.ycal_type + ' [' + self.xunit + ']')

    def plot_vline(self, x, *args, **kwargs):
        '''
        Plots a vertical line on the 2D spectrum
        '''
        plt.vlines(x, self.ycal.min(), self.ycal.max(), *args, **kwargs)

    def plot_hline(self, y, *args, **kwargs):
        '''
        Plots a horizontal line on 2D spectrum
        '''
        plt.hlines(y, self.xcal.min(), self.xcal.max(), *args, **kwargs)

    def plot_roi(self, xroi, yroi, *args, **kwargs):
        '''
        Plots a roi on the 2D spectrum
        '''
        plt.plot([self.xcal[xroi[0]], self.xcal[xroi[0]]], [self.ycal[yroi[0]], self.ycal[yroi[1]]], *args, **kwargs)
        plt.plot([self.xcal[xroi[1]], self.xcal[xroi[1]]], [self.ycal[yroi[0]], self.ycal[yroi[1]]], *args, **kwargs)
        plt.plot([self.xcal[xroi[0]], self.xcal[xroi[1]]], [self.ycal[yroi[0]], self.ycal[yroi[0]]], *args, **kwargs)
        plt.plot([self.xcal[xroi[0]], self.xcal[xroi[1]]], [self.ycal[yroi[1]], self.ycal[yroi[1]]], *args, **kwargs)

    def plot_ellipse(self, x, y, width, height, *args, **kwargs):
        ellipse = patches.Ellipse((x, y), width, height, *args, **kwargs)
        ax = plt.gca()
        ax.add_patch(ellipse)


class Xslice():
    def __init__(self, slice_lim=None, slice_mean=False,
                 counts=None, err=None, ndat=None, counts_type=[None, None],
                 xcal=None, xcal_type=None, xunit=None, vunit=None,
                 C2R=None):

        self.counts = counts
        self.err = err
        self.ndat = ndat
        self.counts_type = counts_type
        self.slice_lim = slice_lim
        if slice_lim is not None:
            self.nlim = slice_lim[1] - slice_lim[0]
        self.slice_mean = False
        self.xcal = xcal
        self.xcal_type = xcal_type
        self.xunit = xunit
        self.vunit = vunit
        self.C2R = C2R

    ## Object generator ##
    @classmethod
    def genobj(cls, imgobj, slice_lim, slice_mean=False):
        if imgobj.counts_type == 'total':
            counts_tmp = cls.__get_slice(imgobj.counts, slice_lim, slice_mean)
            err_tmp = cls.__get_slice_err(imgobj.counts, slice_lim, slice_mean)
            if slice_mean:
                counts_type_tmp = [imgobj.counts_type, 'mean']
            else:
                counts_type_tmp = [imgobj.counts_type, 'total']
            vunit_tmp = cls.__get_vunit(counts_type_tmp)
            return cls(slice_lim=slice_lim,
                          counts=counts_tmp, err=err_tmp, ndat=imgobj.ndat, counts_type=counts_type_tmp,
                          xcal=imgobj.xcal, xcal_type=imgobj.xcal_type, xunit=imgobj.xunit, vunit=vunit_tmp,
                          C2R=imgobj.C2R)
        else:
            raise ValueError('---- input imgobj.counts_type should be "total" ----')

    ## Class helpers ##
    @staticmethod
    def __get_vunit(counts_type):
        if counts_type == ['total', 'total']:
            vunit = '#'
        elif counts_type == ['total', 'mean']:
            vunit = '#/pix'
        elif counts_type == ['mean', 'total']:
            vunit = '#/min'
        elif counts_type == ['mean', 'mean']:
            vunit = '#/min/pix'
        return vunit

    @staticmethod
    def __get_slice(counts, slice_lim, slice_mean=False):

        if np.size(slice_lim) == 1:
            print(slice_lim)
            return counts[slice_lim, :]

        else:
            if slice_mean:
                return np.nanmean(counts[slice_lim[0]:slice_lim[1], :], axis=0)

            else:
                return np.nansum(counts[slice_lim[0]:slice_lim[1], :], axis=0)

    @staticmethod
    def __get_slice_err(counts, slice_lim, slice_mean=False):

        if np.size(slice_lim) == 1:
            return np.sqrt(counts[slice_lim, :])
        else:
            err_tmp = np.sqrt(np.nansum(counts[slice_lim[0]:slice_lim[1], :], axis=0))
            if slice_mean:
                return err_tmp/(slice_lim[1] - slice_lim[0])
            else:
                return err_tmp

    ## Data processors ##
    def mean(self):
        if self.counts_type[0] == 'total':
            counts_mean = self.counts/self.ndat
            err_mean = self.err/self.ndat
            counts_type_tmp = ['mean', self.counts_type[1]]
            vunit_tmp = self.__get_vunit(counts_type_tmp)
            return Xslice(slice_lim=self.slice_lim,
                          counts=counts_mean, err=err_mean, ndat=self.ndat, counts_type=counts_type_tmp,
                          xcal=self.xcal, xcal_type=self.xcal_type, xunit=self.xunit, vunit=vunit_tmp,
                          C2R=self.C2R)
        else:
            raise ValueError('---- self.counts_type[0] should be "total" ----')

    def sub(self, other):
        if (self.counts_type == ['mean', 'mean'] and other.counts_type == ['mean', 'mean']) or (self.counts_type == ['mean', 'total'] and other.counts_type == ['mean', 'total']):
            counts_sub = self.counts - other.counts
            err_sub = np.sqrt(self.err**2 + other.err**2)
            self.counts_type.append('sub')
            return Xslice(slice_lim=self.slice_lim,
                          counts=counts_sub, err=err_sub, ndat=1, counts_type=self.counts_type,
                          xcal=self.xcal, xcal_type=self.xcal_type, xunit=self.xunit, vunit=self.vunit,
                          C2R=self.C2R)
        else:
            raise ValueError('---- both counts_type should be ["mean", "mean"] ----')

    def cnts2ray(self):
        if self.counts_type[0] == 'mean' and self.counts_type[1] == 'mean':
            self.rayleigh = self.counts * self.C2R
            self.rayleigh_err = self.err * self.C2R
        else:
            raise ValueError('---- self.counts_type should be ["mean", "mean" (, "sub")] ----')

    ## Brighntess calculator ##
    def get_brightness(self, xlim, npix=None):
        if npix is None:
            if self.counts_type == ['mean', 'mean'] or self.counts_type == ['mean', 'mean', 'sub']:
                brightness = np.nansum(self.rayleigh[xlim[0]:xlim[1]])
                brightness_err = np.sqrt(np.nansum(self.rayleigh_err[xlim[0]:xlim[1]]**2)) ## it is correct if assuming C2R is almost constant
            else:
                raise ValueError('---- counts_type should be ["mean", "mean" (, "sub")] ----')
        else:
            if self.counts_type == ['mean', 'total', 'sub']:
                brightness = np.nansum(self.counts[xlim[0]:xlim[1]])/npix*np.nanmean(self.C2R[xlim[0]:xlim[1]])
                brightness_err = np.sqrt(np.nansum(self.err[xlim[0]:xlim[1]]**2))/npix*np.nanmean(self.C2R[xlim[0]:xlim[1]]) ## it is correct if assuming C2R is almost constant

            else:
                raise ValueError('---- counts_type should be ["mean", "total"] or "sub"----')
        return brightness, brightness_err

    def ang2nm(self):
        if self.xcal_type == 'wavelength' and self.xunit == 'Å':
            self.xcal = self.xcal * 0.1
            self.xunit = 'nm'

    def nm2ang(self):
        if self.xcal_type == 'wavelength' and self.xunit == 'nm':
            self.xcal = self.xcal * 10
            self.xunit = 'Å'

    ## Visualization methods ##
    def plot(self, rayleigh=False, *args, **kwargs):

        if rayleigh:
            plt.errorbar(self.xcal, self.rayleigh, self.err, drawstyle='steps-mid', *args, **kwargs)
            if self.xcal_type == 'bin':
                plt.xlabel('spectral bin')
            else:
                plt.xlabel(self.xcal_type + ' [' + self.xunit + ']')
            plt.ylabel('Rayleigh/pix')
        else:
            plt.errorbar(self.xcal, self.counts, self.err, drawstyle='steps-mid', *args, **kwargs)
            if self.xcal_type == 'bin':
                plt.xlabel('spectral bin')
            else:
                plt.xlabel(self.xcal_type + ' [' + self.xunit + ']')
            plt.ylabel(self.vunit)

    def plot_vline(self, x, *args, **kwargs):
        '''
        Plots a vertical line on the 2D spectrum
        '''
        vlim = plt.gca().get_ylim()
        plt.vlines(x, vlim[0], vlim[1], *args, **kwargs)


    def plot_hline(self, y, *args, **kwargs):
        '''
        Plots a horizontal line on 2D spectrum
        '''
        hlim = plt.gca().get_xlim()
        plt.hlines(y, hlim[0], hlim[1], *args, **kwargs)

class Yslice():
    def __init__(self, slice_lim=None, slice_mean=False,
                 counts=None, err=None, ndat=None, counts_type=[None, None],
                 ycal=None, ycal_type=None, yunit=None, vunit=None,
                 C2R=None):

        self.counts = counts
        self.err = err
        self.ndat = ndat
        self.counts_type = counts_type
        self.slice_lim = slice_lim
        if slice_lim is not None:
            self.nlim = slice_lim[1] - slice_lim[0]
        self.slice_mean = False
        self.ycal = ycal
        self.ycal_type = ycal_type
        self.yunit = yunit
        self.vunit = vunit
        self.C2R = C2R

    ## Object generator ##
    @classmethod
    def genobj(cls, imgobj, slice_lim, slice_mean=False):
        if imgobj.counts_type == 'total':
            counts_tmp = cls.__get_slice(imgobj.counts, slice_lim, slice_mean)
            err_tmp = cls.__get_slice_err(imgobj.counts, slice_lim, slice_mean)

            if slice_mean:
                counts_type_tmp = [imgobj.counts_type, 'mean']
            else:
                counts_type_tmp = [imgobj.counts_type, 'total']
            vunit_tmp = cls.__get_vunit(counts_type_tmp)
            return cls(slice_lim=slice_lim,
                          counts=counts_tmp, err=err_tmp, ndat=imgobj.ndat, counts_type=counts_type_tmp,
                          ycal=imgobj.ycal, ycal_type=imgobj.ycal_type, yunit=imgobj.yunit, vunit=vunit_tmp,
                          C2R=imgobj.C2R)
        else:
            raise ValueError('---- input imgobj.counts_type should be "total" ----')

    ## Class helpers ##
    @staticmethod
    def __get_vunit(counts_type):
        if counts_type == ['total', 'total']:
            vunit = '#'
        elif counts_type == ['total', 'mean']:
            vunit = '#/pix'
        elif counts_type == ['mean', 'total']:
            vunit = '#/min'
        elif counts_type == ['mean', 'mean']:
            vunit = '#/min/pix'
        return vunit
    @staticmethod
    def __get_slice(counts, slice_lim, slice_mean=False):

        if np.size(slice_lim) == 1:
            return counts[:, slice_lim]

        else:
            if slice_mean:
                return np.nanmean(counts[:, slice_lim[0]:slice_lim[1]], axis=1)

            else:
                return np.nansum(counts[:, slice_lim[0]:slice_lim[1]], axis=1)
    @staticmethod
    def __get_slice_err(counts, slice_lim, slice_mean=False):

        if np.size(slice_lim) == 1:
            return np.sqrt(counts[:, slice_lim])
        else:
            err_tmp = np.sqrt(np.nansum(counts[:, slice_lim[0]:slice_lim[1]], axis=1))
            if slice_mean:
                return err_tmp/(slice_lim[1] - slice_lim[0])
            else:
                return err_tmp

    ## Data processors ##
    def mean(self):
        if self.counts_type[0] == 'total':
            counts_mean = self.counts/self.ndat
            err_mean = self.err/self.ndat
            counts_type_tmp = ['mean', self.counts_type[1]]
            vunit_tmp = self.__get_vunit(counts_type_tmp)
            return Yslice(slice_lim=self.slice_lim,
                          counts=counts_mean, err=err_mean, ndat=self.ndat, counts_type=counts_type_tmp,
                          ycal=self.ycal, ycal_type=self.ycal_type, yunit=self.yunit, vunit=vunit_tmp,
                          C2R=self.C2R)
        else:
            raise ValueError('---- self.counts_type[0] should be "total" ----')

    def sub(self, other):
        if (self.counts_type == ['mean', 'mean'] and other.counts_type == ['mean', 'mean']) or (self.counts_type == ['mean', 'total'] and other.counts_type == ['mean', 'total']):
            counts_sub = self.counts - other.counts
            err_sub = np.sqrt(self.err**2 + other.err**2)
            self.counts_type.append('sub')
            return Yslice(slice_lim=self.slice_lim,
                          counts=counts_sub, err=err_sub, ndat=1, counts_type=self.counts_type,
                          ycal=self.ycal, ycal_type=self.ycal_type, yunit=self.yunit, vunit=self.vunit,
                          C2R=self.C2R)
        else:
            raise ValueError('---- both counts_type should be ["mean", "mean"] ----')

    def adjust_bg(self, bg, ylim):
        if self.counts_type == ['total', 'total'] and bg.counts_type == ['total', 'total'] :
            c_on = np.sum(self.counts[ylim[0]:ylim[1]])
            c_on_err = np.sqrt(c_on)
            c_bg = np.sum(bg.counts[ylim[0]:ylim[1]])
            c_bg_err = np.sqrt(c_bg)
            f = c_on/c_bg
            f_err = np.sqrt((c_on_err/c_bg)**2 + (c_on*c_bg_err/c_bg**2)**2)

            counts_new = bg.counts * f
            err_new = np.sqrt((f*bg.err)**2 + (bg.counts*f_err)**2)

            return Yslice(slice_lim=self.slice_lim,
                      counts=counts_new, err=err_new, ndat=self.ndat, counts_type=self.counts_type,
                      ycal=self.ycal, ycal_type=self.ycal_type, yunit=self.yunit, vunit=self.vunit,
                      C2R=self.C2R)
        else:
            raise ValueError('---- both counts_type should be ["total", "total"] ----')

    def cnts2ray(self):
        if self.counts_type == ['mean', 'mean'] or self.counts_type == 'sub':
            self.rayleigh = self.counts * np.nanmean(self.C2R[self.slice_lim[0]:self.slice_lim[1]])
            self.rayleigh_err = self.err * np.nanmean(self.C2R[self.slice_lim[0]:self.slice_lim[1]])
        else:
            raise ValueError('---- self.counts_type should be ["mean", "mean"] ----')

    ## Brighntess calculator ##
    def get_brightness(self, ylim, npix=None):
        if npix is None:
            if self.counts_type == ['mean', 'mean'] or self.counts_type == ['mean', 'mean', 'sub']:
                brightness = np.nanmean(self.counts[ylim[0]:ylim[1]]) * np.nanmean(self.C2R[self.slice_lim[0]:self.slice_lim[1]])
                brightness_err = np.sqrt(np.nansum(self.err[ylim[0]:ylim[1]]**2))/(ylim[1]-ylim[0]) * np.nanmean(self.C2R[self.slice_lim[0]:self.slice_lim[1]])## it is correct if assuming C2R is almost constant
            else:
                raise ValueError('---- counts_type should be ["mean", "mean" (, "sub")] ----')
        else:
            if self.counts_type == ['mean', 'total', 'sub']:
                brightness = np.nansum(self.counts[ylim[0]:ylim[1]])/npix*np.nanmean(self.C2R[self.slice_lim[0]:self.slice_lim[1]])
                brightness_err = np.sqrt(np.nansum(self.err[ylim[0]:ylim[1]]**2))/npix*np.nanmean(self.C2R[self.slice_lim[0]:self.slice_lim[1]]) ## it is correct if assuming C2R is almost constant

            else:
                raise ValueError('---- counts_type should be ["mean", "total" (, "sub")] ----')
        return brightness, brightness_err

    ## Visualization methods ##
    def plot(self, rayleigh=False, errorbar=True, *args, **kwargs):

        if rayleigh:
            if errorbar:
                plt.errorbar(self.ycal, self.rayleigh, self.rayleigh_err, drawstyle='steps-mid', *args, **kwargs)
            else:
                plt.step(self.ycal, self.rayleigh, where='mid', *args, **kwargs)
            if self.ycal_type == 'bin':
                plt.xlabel('Spatial bin')
            else:
                plt.xlabel(self.ycal_type + ' [' + self.yunit + ']')
            plt.ylabel('Rayleigh/pix')
        else:
            if errorbar:
                plt.errorbar(self.ycal, self.counts, self.err, drawstyle='steps-mid', *args, **kwargs)
            else:
                plt.step(self.ycal, self.counts, where='mid', *args, **kwargs)
            if self.ycal_type == 'bin':
                plt.xlabel('spatial bin')
            else:
                plt.xlabel(self.ycal_type + ' [' + self.yunit + ']')
            plt.ylabel(self.vunit)

    def plot_vline(self, x, *args, **kwargs):
        '''
        Plots a vertical line on the 2D spectrum
        '''
        vlim = plt.gca().get_ylim()
        plt.vlines(x, vlim[0], vlim[1], *args, **kwargs)


    def plot_hline(self, y, *args, **kwargs):
        '''
        Plots a horizontal line on 2D spectrum
        '''
        hlim = plt.gca().get_xlim()
        plt.hlines(y, hlim[0], hlim[1], *args, **kwargs)


class DiskBrightness:
    def __init__(self, img, ndat, img_sky, ndat_sky, xlim, ylim, ylim_adj=None):
        self.img = img
        self.ndat = ndat
        self.img_sky = img_sky
        self.ndat_sky = ndat_sky
        self.xlim = xlim
        self.ylim = ylim
        self.ylim_adj = ylim_adj
        self.img_err = np.sqrt(img)
        self.img_sky_err = np.sqrt(img_sky)
        self.xscl = 10
        self.yscl = 4.2

        dic = get_summary(self.img, self.ndat,
                          self.img_sky, self.ndat_sky,
                          self.xlim, self.ylim, self.ylim_adj)
        self.img_mean = dic['img_mean']
        self.img_sky_mean = dic['img_sky_mean']
        self.img_sub = dic['img_sub']
        self.img_mean_err = dic['img_mean_err']
        self.img_sky_mean_err = dic['img_sky_mean_err']
        self.img_sub_err = dic['img_sub_err']
        self.xslice_mean = dic['xslice_mean']
        self.xslice_sky_mean = dic['xslice_sky_mean']
        self.xslice_sub = dic['xslice_sub']
        self.xslice_mean_err = dic['xslice_mean_err']
        self.xslice_sky_mean_err = dic['xslice_sky_mean_err']
        self.xslice_sub_err = dic['xslice_sub_err']
        self.yslice_mean = dic['yslice_mean']
        self.yslice_sky_mean = dic['yslice_sky_mean']
        self.yslice_sub = dic['yslice_sub']
        self.yslice_mean_err = dic['yslice_mean_err']
        self.yslice_sky_mean_err = dic['yslice_sky_mean_err']
        self.yslice_sub_err = dic['yslice_sub_err']
        self.f_adj = dic['f_adj']
        self.unit = 'counts/min'
        self.target = None

    def add_target(self, target):
        self.target = target

    def add_appdia(self, appdia_asec, factor=1):
        '''
        add apparent diameter in arcsec
        '''
        self.appdia = appdia_asec * factor # (Do multiply a factor that affects brightness value)
        self.appdia_wv = appdia_asec/self.xscl  # (Do not multiply a factor)
        self.appdia_ysl = appdia_asec/self.yscl # (Do not multiply a factor)

    def add_illu_fra(self, f_illu):
        '''
        add illumination fraction (0<=f<=1.0)
        '''
        f_illu = f_illu/100 if f_illu > 1 else f_illu
        self.f_illu = f_illu

    def calc_correction_factor(self, disk_factor=1.):
        self.disk_factor = disk_factor
        omega_p = asec2rad(self.yscl) * asec2rad(self.xscl)
        omega_d = get_omega(self.appdia*self.disk_factor) * self.f_illu
        omega_d_half = omega_d * 0.5 #get_omega(self.appdia/2*self.disk_factor) * self.f_illu
        #print(self.appdia)
        self.omega_ratio = omega_p/omega_d
        self.omega_ratio_half = omega_p/omega_d_half
        #print(self.omega_ratio)
        #print(4.2 * 10/np.pi/(self.appdia/2)**2/self.f_illu)

    def cnts2ray(self):
        self.img_mean = self.img_mean * self.C2Rtbl
        self.img_sky_mean = self.img_sky_mean * self.C2Rtbl
        self.img_sub = self.img_sub * self.C2Rtbl
        self.xslice_mean = self.xslice_mean * self.C2R
        self.xslice_sky_mean = self.xslice_sky_mean * self.C2R
        self.xslice_sub = self.xslice_sub * self.C2R
        self.xslice_mean_err = self.xslice_mean_err * self.C2R
        self.xslice_sky_mean_err = self.xslice_sky_mean_err * self.C2R
        self.xslice_sub_err = self.xslice_sub_err * self.C2R
        self.yslice_mean = self.yslice_mean * self.C2R
        self.yslice_sky_mean = self.yslice_sky_mean * self.C2R
        self.yslice_sub = self.yslice_sub * self.C2R
        self.yslice_mean_err = self.yslice_mean_err * self.C2R
        self.yslice_sky_mean_err = self.yslice_sky_mean_err * self.C2R
        self.yslice_sub_err = self.yslice_sub_err * self.C2R
        self.unit = 'Rayleigh'

    def add_C2R_table(self, C2Rtbl):
        self.C2Rtbl = C2Rtbl
        self.C2R = C2Rtbl[550]

    def add_xcal(self, xcal):
        self.xcal = xcal

    def add_ycal(self, ycal=np.arange(1024)):
        self.ycal = ycal

    def calc_brightness(self):
        if self.unit == 'counts/min':
            counts_tot = np.sum(self.img_sub[self.ylim[0]:self.ylim[1],self.xlim[0]:self.xlim[1]])
            counts_err = np.sqrt(np.sum(self.img_sub_err[self.ylim[0]:self.ylim[1],self.xlim[0]:self.xlim[1]]**2))
            C2R_avg = np.mean(self.C2R[self.xlim[0]:self.xlim[1]])
            self.disk_brightness = counts_tot*C2R_avg*self.omega_ratio
            self.disk_brightness_err = counts_err*C2R_avg*self.omega_ratio
            #self.img_ray = self.img_sub * self.C2Rtbl
            #self.img_ray_err = self.img_sub * self.C2Rtbl
            #ray = np.sum(self.img_ray[self.ylim[0]:self.ylim[1],self.xlim[0]:self.xlim[1]])
            #print(ray)
            #print(ray * self.omega_ratio)
        else:
            ray = np.sum(self.img_sub[self.ylim[0]:self.ylim[1],self.xlim[0]:self.xlim[1]])
            ray_err = np.sqrt(np.sum(self.img_sub_err[self.ylim[0]:self.ylim[1],self.xlim[0]:self.xlim[1]]**2))
            self.disk_brightness = np.array(ray)*self.omega_ratio
            self.disk_brightness_err = np.array(ray_err)*self.omega_ratio

    def calc_brightness_half_dist(self, ylim_dist):
        if self.unit == 'counts/min':
            counts_tot = np.sum(self.img_sub[ylim_dist[0]:ylim_dist[1],self.xlim[0]:self.xlim[1]])
            counts_err = np.sqrt(np.sum(self.img_sub_err[ylim_dist[0]:ylim_dist[1],self.xlim[0]:self.xlim[1]]**2))
            C2R_avg = np.mean(self.C2R[self.xlim[0]:self.xlim[1]])
            self.disk_brightness = counts_tot*C2R_avg*self.omega_ratio_half
            self.disk_brightness_err = counts_err*C2R_avg*self.omega_ratio_half
            #self.img_ray = self.img_sub * self.C2Rtbl
            #self.img_ray_err = self.img_sub * self.C2Rtbl
            #ray = np.sum(self.img_ray[ylim_dist[0]:ylim_dist[1],self.xlim[0]:self.xlim[1]])
            #print(ray)
            #print(ray * self.omega_ratio)
        else:
            ray = np.sum(self.img_sub[ylim_dist[0]:ylim_dist[1],self.xlim[0]:self.xlim[1]])
            ray_err = np.sqrt(np.sum(self.img_sub_err[ylim_dist[0]:ylim_dist[1],self.xlim[0]:self.xlim[1]]**2))
            self.disk_brightness = np.array(ray)*self.omega_ratio_half
            self.disk_brightness_err = np.array(ray_err)*self.omega_ratio_half

    def summary_plot(self, figsize=(12,8)):

        """imgobj = Img().genobj(self.img, self.ndat)
        imgobj_sky = Img().genobj(self.img_sky, self.ndat_sky)

        imgobj_mean = imgobj.mean()
        imgobj_sky_mean = imgobj_sky.mean()
        xslice = Xslice().genobj(imgobj, self.ylim)
        xslice_mean = xlsice.mean()
        yslice = Yslice().genobj(imgobj, self.xlim)
        yslice_mean = yslice.mean()

        if self.adjust_sky_level:
            imgobj_sky_adj = imgobj_sky_mean.adjust_bg(imgobj_mean, self.xlim, self.ylim)
            xslice_sky_adj = Xslice().genobj(imgobj_sky_adj, self.ylim)
            yslice_sky_adj = Yslice().genobj(imgobj_sky_adj, self.xlim)
            img_sub = imgobj_mean.sub(imgobj_sky_adj)
            xslice_sub = xslice_mean.sub(xslice_sky_adj)
            yslice_sub = yslice.sub(yslice_sky_adj)
        else:
            xslice_mean = Xslice().genobj(imgobj_mean, self.ylim)
            xslice_sky_mean = Xslice().genobj(imgobj_sky_mean, self.ylim)
            yslice_mean = Yslice().genobj(imgobj_mean, self.xlim)
            yslice_sky_mean = Yslice().genobj(imgobj_sky_mean, self.xlim)
            img_sub = imgobj.sub(imgobj_sky)
            xslice_sub = xslice_mean.sub(xslice_sky_mean)
            yslice_sub = yslice_mean.sub(yslice_sky_mean)"""

        """dic = get_summary(self.img, self.img_sky, self.ndat, self.ndat_sky, self.xlim, self.ylim, self.ylim_adj)
        img_mean = dic['img_mean']
        img_sky_mean = dic['img_sky_mean']
        img_sub = dic['img_sub']

        self.xslice_mean = dic['xslice_mean']
        self.xslice_sky_mean = dic['xslice_sky_mean']
        self.xslice_sub = dic['xslice_sub']
        self.xslice_mean_err = dic['xslice_mean_err']
        self.xslice_sky_mean_err = dic['xslice_sky_mean_err']
        self.xslice_sub_err = dic['xslice_sub_err']

        self.yslice_mean = dic['yslice_mean']
        self.yslice_sky_mean = dic['yslice_sky_mean']
        self.yslice_sub = dic['yslice_sub']
        self.yslice_mean_err = dic['yslice_mean_err']
        self.yslice_sky_mean_err = dic['yslice_sky_mean_err']
        self.yslice_sub_err = dic['yslice_sub_err']"""

        # Start plot
        #settings
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=figsize)
        widths = [3, 1]
        gs = fig.add_gridspec(5, 2, width_ratios=widths)#fig.add_gridspec(5, 2, height_ratios=heights)
        plt.subplots_adjust(hspace = 0.75)
        wvlim = [self.xcal[self.xlim[1]], self.xcal[self.xlim[0]]]
        shifted_x, shifted_y = shift_grids(self.xcal, self.ycal)
        vmax_img = np.abs(np.max(self.img_mean[self.ylim[0]:self.ylim[1], self.xlim[0]:self.xlim[1]]))
        vmax_xslice = np.max(self.xslice_mean[self.xlim[0]:self.xlim[1]])*1.1
        vmax_yslice = np.max(self.yslice_mean[self.ylim[0]:self.ylim[1]])*1.1

        #panel0
        ax0 = fig.add_subplot(gs[0,0])
        ax0.set_title('target:' + str(self.ndat) + ' min') if self.target is None else ax0.set_title(self.target + ': ' + str(self.ndat) + ' min')
        ax0.set_xlim(wvlim[0]-50, wvlim[1]+50)
        ax0.set_ylim(490, 650)
        mesh0 = ax0.pcolormesh(shifted_x, shifted_y, self.img_mean, vmin=0, vmax=vmax_img, rasterized=True)
        cb0 = plt.colorbar(mesh0)
        cb0.set_label(self.unit, rotation=270, labelpad=10)

        ax1 = fig.add_subplot(gs[1,0])
        ax1.set_title('sky: ' + str(self.ndat_sky) + ' min') if self.target is None else ax1.set_title(self.target + '_sky: ' + str(self.ndat_sky) + ' min')
        ax1.set_xlim(wvlim[0]-50, wvlim[1]+50)
        ax1.set_ylim(490, 650)
        mesh1 = ax1.pcolormesh(shifted_x, shifted_y, self.img_sky_mean, vmin=0, vmax=vmax_img, rasterized=True)
        cb1 = plt.colorbar(mesh1)
        cb1.set_label(self.unit, rotation=270, labelpad=10)

        ax2 = fig.add_subplot(gs[2,0])
        ax2.set_title('Diff')
        ax2.set_xlim(wvlim[0]-50, wvlim[1]+50)
        ax2.set_ylim(490, 650)
        mesh2 = ax2.pcolormesh(shifted_x, shifted_y, self.img_sub, cmap='RdYlBu_r', vmin=-vmax_img, vmax=vmax_img, rasterized=True)
        cb2 = plt.colorbar(mesh2)
        cb2.set_label(self.unit, rotation=270, labelpad=10)

        ax3 = fig.add_subplot(gs[0,1])
        ax3.set_xlim(wvlim)
        ax3.set_ylim(self.ylim)
        mesh3 = ax3.pcolormesh(shifted_x, shifted_y, self.img_mean, vmin=0, vmax=vmax_img, rasterized=True)
        cb3 = plt.colorbar(mesh3)
        cb3.set_label(self.unit, rotation=270, labelpad=10)

        ax4 = fig.add_subplot(gs[1,1])
        ax4.set_xlim(wvlim)
        ax4.set_ylim(self.ylim)
        mesh4 = ax4.pcolormesh(shifted_x, shifted_y, self.img_sky_mean, vmin=0, vmax=vmax_img, rasterized=True)
        cb4 = plt.colorbar(mesh4)
        cb4.set_label(self.unit, rotation=270, labelpad=10)

        ax5 = fig.add_subplot(gs[2,1])
        ax5.set_xlim(wvlim)
        ax5.set_ylim(self.ylim)
        mesh5 = ax5.pcolormesh(shifted_x, shifted_y, self.img_sub, cmap='RdYlBu_r', vmin=-vmax_img, vmax=vmax_img, rasterized=True)
        cb5 = plt.colorbar(mesh5)
        cb5.set_label(self.unit, rotation=270, labelpad=10)


        ## Line plots
        ax6 = fig.add_subplot(gs[3,:])
        ax6.set_xlim(wvlim[0]-50, wvlim[1]+50)
        ax6.set_ylim(-vmax_xslice*0.2, vmax_xslice)
        ax6.set_xlabel('wavelength [Å]')
        ax6.axhline(0, color='lightgray', linestyle='dashed')
        ax6.errorbar(self.xcal, self.xslice_mean, self.xslice_mean_err, color='C3', ds='steps-mid')
        ax6.errorbar(self.xcal, self.xslice_sky_mean, self.xslice_sky_mean_err, color='C0', ds='steps-mid')
        ax6.errorbar(self.xcal, self.xslice_sub, self.xslice_sub_err, color='k', ds='steps-mid')
        ax6.axvline(wvlim[0], linestyle='--', color='C3')
        ax6.axvline(wvlim[1], linestyle='--', color='C3')
        ## Putting below code makes plotting very slow!! Do not use unless really needed.
        #obx = AnchoredHScaleBar(size=self.appdia_wv, label="2 ${R_{M}}$", loc='lower center', frameon=False, pad=0,sep=4,color="k", linewidth=0.8, ax=ax6)
        #ax6.add_artist(obx)

        ax7 = fig.add_subplot(gs[4,:])
        ax7.set_xlim(490, 650)
        ax7.set_ylim(-vmax_yslice*0.2, vmax_yslice)
        ax7.set_xlabel('spatial bins')
        ax7.axhline(0, color='lightgray', linestyle='dashed')
        ax7.errorbar(self.ycal, self.yslice_mean, self.yslice_mean_err, color='C3', ds='steps-mid')
        ax7.errorbar(self.ycal, self.yslice_sky_mean, self.yslice_sky_mean_err, color='C0', ds='steps-mid')
        ax7.errorbar(self.ycal, self.yslice_sub, self.yslice_sub_err, color='k', ds='steps-mid')
        ax7.axvline(self.ylim[0], linestyle='--', color='C3')
        ax7.axvline(self.ylim[1], linestyle='--', color='C3')
        ax7.text(625, vmax_yslice*0.7, '{:.2f}'.format(self.disk_brightness)+'±'+'{:.2f}'.format(self.disk_brightness_err)+' R')
        ## Putting below code makes plotting very slow!! Do not use unless really needed.
        #oby = AnchoredHScaleBar(size=self.appdia_ysl, label="2 ${R_{M}}$", loc='lower center', frameon=False, pad=0,sep=4,color="k", linewidth=0.8, ax=ax7)
        #ax7.add_artist(oby)

        [iax.set_xlabel('wavelength [Å]') for iax in [ax0, ax1, ax2, ax3, ax4, ax5]]
        [iax.set_ylabel('spatial bins') for iax in [ax0, ax1, ax2, ax3, ax4, ax5]]
        [iax.set_ylabel(self.unit) for iax in [ax6, ax7]]
        [iax.axvline(wvlim[0], color='r', linestyle='--') for iax in [ax0, ax1, ax2, ax3, ax4, ax5]]
        [iax.axvline(wvlim[1], color='r', linestyle='--') for iax in [ax0, ax1, ax2, ax3, ax4, ax5]]
        [iax.axhline(self.ylim[0], color='r', linestyle='--') for iax in [ax0, ax1, ax2, ax3, ax4, ax5]]
        [iax.axhline(self.ylim[1], color='r', linestyle='--') for iax in [ax0, ax1, ax2, ax3, ax4, ax5]]

class HskRoi():
    '''
    HskRoi object can get region of interests (roi) in the hisaki spectrum.
    '''
    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim
        if xlim is None and ylim is None:
            self.xcal = None
            self.xlim_wl = None
            self.ycal = None
        else:
            self.xcal, _, _ = get_cal()
            self.xlim_wl = [self.xcal[self.xlim[0]], self.xcal[self.xlim[1]]]
            self.xlim_wl.sort()
            self.ycal = None

    def get_nroi(self):
        self.nroi = len(range(self.xlim[0], self.xlim[1])) * len(range(self.ylim[0], self.ylim[1]))
        return self.nroi

    def plot(self, wl=True, dist=False, *args, **kwargs):
        if wl == True and dist == False:
            x = self.xlim_wl
            y = self.ylim
        elif wl == False and dist == False:
            x = self.xlim
            y = self.ylim

        plt.plot([x[0],x[0]],[y[0], y[1]], *args, **kwargs)
        plt.plot([x[1],x[1]],[y[0], y[1]], *args, **kwargs)
        plt.plot([x[0],x[1]],[y[0], y[0]], *args, **kwargs)
        plt.plot([x[0],x[1]],[y[1], y[1]], *args, **kwargs)
