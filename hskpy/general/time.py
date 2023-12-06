import numpy as np
from datetime import datetime, timedelta
import calendar

str_format = '%Y-%m-%dT%H:%M:%S' ## Hisaki time string format
#######################
#### Time handling ####
#######################

def str2Dt(str_time):
    '''
    Converts the Hisaki string format to datetime object
    argment:
        str_time: a string time in the Hisaki string format or a list or an array of those
    return:
        timeDt: a datetime or a list of those
    '''
    if np.size(str_time) == 1:
        if type(str_time) is str:
            timeDt = datetime.strptime(str_time, str_format)
        else:
            timeDt = datetime.strptime(str_time[0], str_format)
    else:
        timeDt = [datetime.strptime(istr, str_format) for istr in str_time]
    return timeDt


def str2unix(str_time):
    '''
    Converts the Hisaki string format to unix time
    argment:
        str_time: a string time in the Hisaki string format or list or array of them
    returns:
        unix_time: a double precision unix time or a list of those
    '''

    unix_time = Dt2unix(str2Dt(str_time))
    return unix_time


def Dt2unix(timeDt):
    '''
    Converts datetime to unix time
    argment:
        timeDt: a datetime or a list of those
    returns:
        unix_time: a double precision unix time or a list of those
    '''
    if np.size(timeDt) == 1:
        unix_time = calendar.timegm(timeDt.timetuple())
    else:
        unix_time = [calendar.timegm(iDt.timetuple()) for iDt in timeDt]
    return unix_time


def Dt2str(timeDt, str_format='%Y-%m-%dT%H:%M:%S'):
    '''
    Converts datetime to the Hisaki string format
    argment:
        timeDt: a datetime or a list of those
    keyword:
        str_format: string date format, defaults to '2021-07-07T09:48:00', for exaple.
    returns:
        str_time: a string time in the Hisaki string format or list or array of them
    '''
    if np.size(timeDt) == 1:
        if type(timeDt) is datetime:
            return timeDt.strftime(str_format)
        else:
            return timeDt[0].strftime(str_format)
    else:
        return [iDt.strftime(str_format) for iDt in timeDt]


def unix2Dt(unix_time):
    '''
    Converts unix time to datetime
    argment:
        unix_time: a double precision unix time or a list of those
    returns:
        timeDt: a datetime or a list of those
    '''
    if np.size(unix_time) == 1:
        if type(unix_time) is float:
            return datetime.utcfromtimestamp(unix_time)
        else:
            return datetime.utcfromtimestamp(unix_time[0])
    else:
        return [datetime.utcfromtimestamp(it) for it in unix_time]


def unix2str(unix_time):
    '''
    Converts unix time to the Hisaki string format
    argment:
        unix_time: a double precision unix time or a list of those
    returns:
        str_time: a string time in the Hisaki string format or list or array of them
    '''
    return Dt2str(unix2Dt(unix_time))

def get_timeDt_mean(timeDt):
    if np.size(timeDt) > 1:
        timeDt_mean = unix2Dt(sum(Dt2unix(timeDt))/len(timeDt))
    elif np.size(timeDt) == 1:
        if isinstance(timeDt, (list, np.ndarray)):
            timeDt_mean = timeDt[0]
        else:
            timeDt_mean = timeDt
    elif np.size(timeDt) == 0:
        timeDt_mean = None
    return timeDt_mean

def interpDt(Dt_new, Dt, y):
    utime_new = Dt2unix(Dt_new)
    utime = Dt2unix(Dt)
    y_new = np.interp(utime_new, utime, y)
    return y_new
