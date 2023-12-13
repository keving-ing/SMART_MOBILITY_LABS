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
                     print(formatted_date)
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
       plt.show()


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

        timeSeries_day(city+"_timeSeries",timeSeries['Bookings'+city])