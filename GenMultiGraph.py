import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Tools import *

Bay_station = pd.read_csv('Bay_Area_Bike/2015/201508_station_data.csv')
Bay_status = pd.read_csv('Bay_Area_Bike/2015/201508_status_data.csv')
Bay_trip = pd.read_csv('Bay_Area_Bike/2015/201508_trip_data.csv')
Bay_weather = pd.read_csv('Bay_Area_Bike/2015/201508_weather_data.csv')

Bay_station[['long_X','lat_Y']] = Bay_station.apply(lambda x: millerToXY(x['long'],x['lat']),
                                                    axis=1, result_type="expand")
dic = {Bay_station.iloc[i,0]:i for i in range(len(Bay_station))}
Bay_station['sid'] = Bay_station['station_id'].map(dic)
Bay_status['time'] = pd.to_datetime(Bay_status['time'])
Bay_status.sort_values(by = 'time')
Bay_status['sid'] = Bay_status['station_id'].map(dic)
Bay_trip['Start Date'] = pd.to_datetime(Bay_trip['Start Date'])
Bay_trip['End Date'] = pd.to_datetime(Bay_trip['End Date'])
Bay_trip['Start_sid'] = Bay_trip['Start Terminal'].map(dic)
Bay_trip['End_sid'] = Bay_trip['End Terminal'].map(dic)

AdjMat = gen_AdjMat(Bay_station, 'lat_Y', 'long_X')
IntMat = gen_InteractMat(Bay_trip,Bay_station,'Start_sid', 'End_sid')
bikes_avail_time_series,docks_availtime_series = get_stations_series(Bay_status,70,
                                                      ['bikes_available','docks_available'],
                                                      time_interval='15T')

bikes_avail_time_series = pad_nan(bikes_avail_time_series)
docks_availtime_series = pad_nan(docks_availtime_series)
PearsonMat1 = get_PearsonMat(bikes_avail_time_series)
PearsonMat2 = get_PearsonMat_2(bikes_avail_time_series,docks_availtime_series)
flow_out,flow_in = get_in_out_series(Bay_trip,70,['2014-09-01 00:00:00',
                                  '2015-08-31 00:00:00'],time_interval='1H')
in_in_P=get_PearsonMat(flow_in)
in_out_P=get_PearsonMat_2(flow_in,flow_out)
plt_heatmap(in_in_P,cmap="viridis")