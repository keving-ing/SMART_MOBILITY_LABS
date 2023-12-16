import pymongo as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar
import gmplot
import math
import mne

def day_of_year_to_date(day_of_year, year):
    reference_date = datetime(year, 1, 1)
    target_date = reference_date + timedelta(days=day_of_year - 1)

    return target_date.strftime('%d')

def timeSeries_day(title,Bookings):
       listBookings = list(Bookings)
       DFBookings=pd.DataFrame(listBookings)
       labels=[]
       ticks=[]

       for i in range(DFBookings.shape[0]):
              if i == (DFBookings.shape[0]-1):
                     break
              elif (DFBookings["_id"][i]["hour"]==0) or ((DFBookings["_id"][i+1]["dow"]-DFBookings["_id"][i]["dow"])!=0) and ((DFBookings["_id"][i+1]["dow"] - DFBookings["_id"][i]["dow"])!=1):
                     ticks.append(i)
                     formatted_date = day_of_year_to_date( DFBookings["_id"][i+1]["dow"], 2017)
                     labels.append(formatted_date)

                    #  if formatted_date == '01':
                    #        ticks.append(0)
                    #        labels.append("02")

                    #  if formatted_date == '23':
                    #        ticks.append(0)
                    #        labels.append("24")

       plt.figure()
       plt.xlabel("Days of October")
       plt.ylabel("N. of Bookings")
       plt.title(title)
       plt.plot(DFBookings["totOFbookings"],label="Bookings")
       plt.xticks(ticks=ticks,
                  labels=labels,
                  rotation=-30)
       plt.legend(loc='best')
       plt.grid(True,which="both")
       #plt.show()

       DFBookings.columns = ['time','rentals']

       df = DFBookings.to_csv(path_or_buf = "C:/Users/kevin/Documents/PoliTo/SMART_MOBILITY/"+title+".csv", index = False)

       return df

def add_miss(df):
    
    df_new = pd.DataFrame(columns=['time', 'rental'])
    for i in range(1, len(df), 1):
        if (str(df.index[i] - df.index[i - 1]) != '0 days 01:00:00'):
            s = pd.date_range(df.index[i - 1], df.index[i], freq='1H')
            for j in range(1, len(s) - 1):
                newvalue = df.iloc[i + j - 24].rental +10 * np.random.random_sample() / 100.0
                print('Fitting', i, j, newvalue)
                df_new = df_new.append({'Time': s[j], 'rental': newvalue}, ignore_index = True)
    df_new = df_new.set_index('time')
    df = df.append(df_new)
    df = df.sort_index()

    plt.plot(df)
    plt.title('Filled rentals timeseries of ')
    plt.xlabel('Date')
    plt.ylabel('Number of rentals')
    plt.xticks(rotation=50)
    plt.grid(linestyle = '--', linewidth=0.8)

    plt.show();

    df.plot(title="Rentals in AMSTERDAM")





if __name__ == "__main__":

    client = pm.MongoClient('bigdatadb.polito.it',
                        port=27017,
                        ssl=True,
                        tlsAllowInvalidCertificates=True,
                        username='ictts',
                        password='Ict4SM22!',
                        authSource='carsharing',
                        #authMechanism='SCRAM-SHA-1'
    )
    db = client['carsharing']
    Bookings = db['PermanentBookings']

    cities=["Denver","Amsterdam","Milano"]

    #---------------------------------------------------------------------------------------------

    #STEP_3 - TASK1
    # Setting of the period of time
    date_init = datetime.strptime('2017-10-01T22:00:00','%Y-%m-%dT%H:%M:%S')
    date_finish = datetime.strptime('2017-10-31T23:59:59','%Y-%m-%dT%H:%M:%S')
    #date_init_Denver = datetime.strptime('2017-10-01T06:00:00','%Y-%m-%dT%H:%M:%S')
    #date_finish_Denver = datetime.strptime('2017-10-30T07:59:59','%Y-%m-%dT%H:%M:%S')
    init_unix = (date_init - datetime(1970, 1, 1)).total_seconds()
    finish_unix = (date_finish - datetime(1970, 1, 1)).total_seconds()


    timeSeries = {}
    for city in cities:
        timeSeries['Bookings{0}'.format(city)] = Bookings.aggregate([
        {"$match": {
                    "city":city,
                    "init_time": {"$gte": init_unix, "$lte": finish_unix}
                   }
                },
        {"$project":{
                "duration":{
                        "$subtract":["$final_time","$init_time"]
                },

                "or_de":{
                        "$ne":[{
                                "$arrayElemAt":["$origin_destination.coordinates",0]
                        },
                        {
                                "$arrayElemAt":["$origin_destination.coordinates",1]
                        }
                        ]
                },
        
            "dow":{"$dayOfYear":"$init_date"},
            "hour" : {"$hour":"$init_date"}  }
                    },
                    {"$match": {
                        "duration":{
                                "$gte": 5*60, "$lte":150*60
                        },
                        "or_de": True 
                    }
                    },

        {"$group":{
            "_id": {"dow":"$dow","hour":"$hour"},
            "totOFbookings":{"$sum":1}
            
                }
                },
        {"$sort":{
                "_id":1
            }
                
                }])

        time_series_df = timeSeries_day(city+"_timeSeries",timeSeries['Bookings'+city])

        df_n = pd.read_csv(r"C:/Users/kevin/Documents/PoliTo/SMART_MOBILITY/"+city+".csv", sep=",", parse_dates=[0], infer_datetime_format = True, date_parser=dateparse, index_col = 0)

        add_miss(time_series_df)