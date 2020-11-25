import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

def getdistance(point, city_point):
    '''
    Calculate the distance
    :param point:
    :param city_point:
    :return:
    '''
    return np.sqrt(np.sum((np.array(point) - np.array(city_point)) ** 2))


def millerToXY(lon, lat):
    """
    :param lon: Longitude
    :param lat: Latitude
    :return:
    """
    L = 6381372 * math.pi * 2  # earth circumference
    W = L  # The plane is expanded and the perimeter is treated as the X axis
    H = L / 2  # The Y axis is about half the circumference
    mill = 2.3  # A constant in Miller's projection, ranging from plus or minus 2.3
    x = lon * math.pi / 180  # Converts longitude from degrees to radians
    y = lat * math.pi / 180
    # Converts latitude from degrees to radians
    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # Here is the transformation of Miller projection

    # Here, the radian is converted into the actual distance, and the unit of conversion is km
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return int(round(x)), int(round(y))

def plot_status_time_series(df,stationid,Date_col,val,time=None):
    '''
    this function is used to plot the Bay Status DataFrame

    :param df: dataframe
    :param stationid: int, the idx of station id
    :param Date_ol: str, the name of date column
    :param val: list or str, list values of graph
    :param time: 2 element list, time interval
    :return:
    '''
    if time is not None:
        df[(df['station_id']==stationid)
           &(df[Date_col]>time[0])
           &(df[Date_col]<time[1])].set_index(Date_col)[val].plot()
    else:
        df[(df['station_id'] == stationid)].set_index(Date_col)[val].plot()

def plot_compare_status_time_series(df,stationids,Date_col,val,time=None):
    '''
    this function is used to plot the Bay Status DataFrame

    :param df: dataframe
    :param stationids: list, the idx of station id
    :param Date_col: str, the name of date column
    :param val: list or str, list values of graph
    :param time: 2 element list, time interval
    :return:
    '''
    if time is not None:
        for i in stationids:
            df[(df['station_id']==i)
               &(df[Date_col]>time[0])
               &(df[Date_col]<time[1])].set_index(Date_col)[val].plot()
    else:
        for i in stationids:
            df[(df['station_id'] == i)].set_index(Date_col)[val].plot()


def gen_AdjMat(df, x_name, y_name):
    cnt = len(df)
    ret_mat = np.zeros([cnt, cnt])
    for i in range(cnt):
        for j in range(i):
            ret_mat[j, i] = ret_mat[i, j] = getdistance(df.iloc[i].loc[[x_name, y_name]],
                                                        df.iloc[j].loc[[x_name, y_name]])
    return ret_mat


def gen_InteractMat(df,df_station,Start_name, End_name):
    cnt = len(df)
    ret_mat = np.zeros([len(df_station),len(df_station)])
    for i in range(cnt):
        Start,End = df.iloc[i].loc[[Start_name, End_name]]
        ret_mat[Start,End] += 1
        if i%10000 == 0:
            print('{} lines done'.format(i))
    return ret_mat


def get_stations_series(df,length,cls,time_interval='15T'):
    '''

    :param df:
    :param length:
    :param cls:
    :param time_interval:
    :return:
    '''
    ret_dim = len(cls)
    ret = []
    for i in range(ret_dim):
        ret.append([])
    for i in range(length):
        sdata = df[df['sid']==i].set_index(['time'])
        series = sdata.loc[:,cls].resample(rule = time_interval).mean()
        for dim in range(ret_dim):
            ret[dim].append(list(series.iloc[:,dim]))
    return ret


def pad_nan(ls):
    '''
    fill up nan using forward fill
    :param ls:
    :return:
    '''
    arr = np.array(ls)
    df = pd.DataFrame(arr.T)
    df.fillna(method='ffill', inplace=True)
    arr = np.array(df)
    return arr.T


def get_PearsonMat(arr):
    '''

    :param arr:
    :return:
    '''
    length = len(arr)
    ret_mat = np.zeros([length,length])
    for i in range(len(arr)):
        for j in range(i):
            ret_mat[i,j]=ret_mat[j,i]=pearsonr(arr[i],arr[j])[0]
    return ret_mat


def get_PearsonMat_2(arr1, arr2):
    '''

    :param arr1:
    :param arr2:
    :return:
    '''
    length = len(arr1)
    ret_mat = np.zeros([length,length])
    for i in range(length):
        for j in range(length):
            ret_mat[i,j] = pearsonr(arr1[i],arr2[j])[0]
    return ret_mat


def norm_mat(mat):
    #ret_mat = np.zeros_like(mat)
    D = np.zeros_like(mat)
    I = np.eye(mat.shape[0])
    for i in range(len(mat)):
        D[i,i] = mat[i].sum()
    ret_mat = np.linalg.inv(D) @ mat + I
    return ret_mat


def norm_mat_without_I(mat):
    #ret_mat = np.zeros_like(mat)
    D = np.zeros_like(mat)
    I = np.eye(mat.shape[0])
    for i in range(len(mat)):
        D[i,i] = mat[i].sum()
    ret_mat = np.linalg.inv(D) @ mat
    return ret_mat


def get_in_out_series(df, length, time, time_interval='15T',
                      start_sid = 'Start_sid',end_sid = 'End_sid',
                      start_date = 'Start Date',end_date = 'End Date'):
    '''
    get in and out flow for every station
    :param df:
    :param length:
    :param time:
    :param time_interval:
    :return:
    '''
    ret_out = []
    ret_in = []
    for i in range(length):
        outdata = df[(df[start_sid] == i)
                     & (df[start_date] > time[0])
                     & (df[start_date] < time[1])]
        outdata = outdata.append({start_date: time[0]}, ignore_index=True)
        outdata = outdata.append({start_date: time[1]}, ignore_index=True)
        outdata[start_date] = pd.to_datetime(outdata[start_date])
        # outdata.sort_values(by = 'Start Date')
        outdata.set_index([start_date], inplace=True)

        outseries = outdata.iloc[:, 0].resample(rule=time_interval).count()
        ret_out.append(list(outseries))

        indata = df[(df[end_sid] == i)
                    & (df[end_date] > time[0])
                    & (df[end_date] < time[1])]
        indata = indata.append({end_date: time[0]}, ignore_index=True)
        indata = indata.append({end_date: time[1]}, ignore_index=True)
        indata[end_date] = pd.to_datetime(indata[end_date])
        indata.set_index([end_date], inplace=True)

        inseries = indata.iloc[:, 0].resample(rule=time_interval).count()
        ret_in.append(list(inseries))
    return ret_out, ret_in, inseries.keys()


def plt_heatmap(Mat,cmap="RdBu"):
    fig = plt.figure(figsize=(16, 12))
    ax = sns.heatmap(Mat, cmap=cmap)

    
def build_statrion_df(df,X,Y,ID,set_station, Community_Areas = None):
    ret_df = pd.DataFrame(columns=['ID', 'long', 'lat', 'Community Areas'])
    for e in range(len(set_station)):
        if Community_Areas is None:
            XY = df[df[ID]==e][[X,Y]].values[0]
            ret_df = ret_df.append({'ID':e,'long':XY[0],'lat':XY[1], 'Community_Areas':None},ignore_index=True)
        else:
            XY = df[df[ID]==e][[X,Y,Community_Areas]].values[0]
            ret_df = ret_df.append({'ID':e,'long':XY[0],'lat':XY[1], 'Community Areas':XY[2]},ignore_index=True)
    return ret_df