import numpy as np

from .data import get_xslice, get_yslice

def get_adjust_factor(img1, nimg1, img2, nimg2, xlim, ylim):
    c_on = np.sum(img1[ylim[0]:ylim[1], xlim[0]:xlim[1]])/nimg1
    c_on_err = np.sqrt(c_on)/nimg1
    c_bg = np.sum(img2[ylim[0]:ylim[1], xlim[0]:xlim[1]])/nimg2
    c_bg_err = np.sqrt(c_bg)/nimg2
    f = c_on/c_bg
    f_err = np.sqrt((c_on_err/c_bg)**2 + (c_on*c_bg_err/c_bg**2)**2)
    return f, f_err

def get_calculated_data(img, ndat, img_sky, ndat_sky, xlim, ylim, ylim_adj=None):

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
