import pymongo as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar
import gmplot
import math
import mne
import io
import json
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


def day_of_year_to_date(day_of_year, year):
    reference_date = datetime(year, 1, 1)
    target_date = reference_date + timedelta(days=day_of_year - 1)

    return target_date.strftime('%d')

def timeSeries_day(title, Bookings):
    listBookings = list(Bookings)
    DFBookings = pd.DataFrame(listBookings)
    labels = []
    ticks = []

    for i in range(DFBookings.shape[0] - 1):
        if (DFBookings["_id"][i]["hour"] == 0) or (
                (DFBookings["_id"][i + 1]["dow"] - DFBookings["_id"][i]["dow"]) != 0) and (
                (DFBookings["_id"][i + 1]["dow"] - DFBookings["_id"][i]["dow"]) != 1):
            ticks.append(i)
            formatted_date = day_of_year_to_date(DFBookings["_id"][i + 1]["dow"], 2017)
            labels.append(formatted_date)

    plt.figure()
    plt.xlabel("Days of October")
    plt.ylabel("N. of Bookings")
    plt.title(title)
    plt.plot(DFBookings["totOFbookings"], label="Bookings")
    plt.xticks(ticks=ticks,
               labels=labels,
               rotation=-30)
    plt.legend(loc='best')
    plt.grid(True, which="both")
    #plt.show()

    DFBookings.columns = ['_id', 'totOFbookings']

    df_result = DFBookings.to_csv(index=False)  # Save to CSV without index

    return df_result

def add_miss(df, title):
    dow_hour_combinations = [(dow, hour) for dow in range(275,305) for hour in range(24)]
    new_rows = []

    print(df)
    for dow, hour in dow_hour_combinations:
        # Check if the combination exists in the DataFrame
        if not any(((eval(entry['_id'])['dow'] == dow) and (eval(entry['_id'])['hour'] == hour)) for _, entry in df.iterrows()):
            # If not, create a new row with rentals value set to 0
            new_rows.append({'_id': {'dow': dow, 'hour': hour}, 'totOFbookings': 0})

    # Create a new DataFrame with the new rows
    new_df = pd.DataFrame(new_rows)

    # Concatenate the original DataFrame and the new DataFrame
    df = pd.concat([df, new_df], ignore_index=True)

    print(df)

    df[['dow', 'hour']] = df['_id'].apply(lambda x: pd.Series([x['dow'], x['hour']] if isinstance(x, dict) else x.split(',')))

# Convert the columns to the appropriate data type
    df['dow'] = df['dow'].apply(lambda x: int(x.split(':')[1].strip('}{ ')) if isinstance(x, str) else int(x))
    df['hour'] = df['hour'].apply(lambda x: int(x.split(':')[1].strip('}{ ')) if isinstance(x, str) else int(x))

# Sort the DataFrame by 'dow' and 'hour'
    df = df.sort_values(by=['dow', 'hour']).reset_index(drop=True) 
    
    print(df)


    labels = []
    ticks = []

    for i in range(df.shape[0] - 1):
        if (df["hour"][i] == 0) or (
            (df["dow"][i + 1] - df["dow"][i]) != 0) and (
            (df["dow"][i + 1] - df["dow"][i]) != 1):
                ticks.append(i)
                print(df["dow"][i + 1])
                formatted_date = day_of_year_to_date(int(df["dow"][i + 1]), 2017)
                labels.append(formatted_date)
    plt.figure()
    plt.xlabel("Days of October")
    plt.ylabel("N. of Bookings")
    plt.title(title)
    plt.plot(df["totOFbookings"], label="Bookings")
    plt.xticks(ticks=ticks,
               labels=labels,
               rotation=-30)
    plt.legend(loc='best')
    plt.grid(True, which="both")
   # plt.show()

    
#     plt.plot(df['totOFbookings'])
#     plt.title('Filled rentals timeseries')
#     plt.xlabel('Index')
#     plt.ylabel('Number of rentals')
#     plt.xticks(rotation=50
#                )
#     plt.grid(linestyle='--', linewidth=0.8)
#     plt.show()
    
    

    #df.plot(y='totOFbookings', title="Rentals in AMSTERDAM")

    return df



def stationarity(df,title):
    labels = []
    ticks = []
    for i in range(df.shape[0] - 1):
        if (df["hour"][i] == 0) or (
            (df["dow"][i + 1] - df["dow"][i]) != 0) and (
            (df["dow"][i + 1] - df["dow"][i]) != 1):
                ticks.append(i)
                print(df["dow"][i + 1])
                formatted_date = day_of_year_to_date(int(df["dow"][i + 1]), 2017)
                labels.append(formatted_date)
    
    df['MA'] = df['totOFbookings'].rolling(24*7).mean() # Moving average 
    df['MS'] = df['totOFbookings'].rolling(24*7).std() # Moving std 
    plt.figure(constrained_layout=True)
    plt.plot(df['totOFbookings'], linewidth=1, label='Number of rentals') 
    plt.plot(df['MA'], linewidth=2, color='r', label='Moving Average') 
    plt.plot(df['MS'], linewidth=2, label='Moving Std')
    plt.title('Moving Average/Std of bookings of '+title)
    plt.xlabel('Date')
    plt.ylabel('Number of bookings')
    plt.xticks(ticks=ticks,
               labels=labels,
               rotation=-30)
    plt.grid(linestyle = '--', linewidth=0.8)
    plt.legend()
    plt.show()

    return df

def autocorrelation(df,title):
     plt.figure(constrained_layout=True)
     pd.plotting.autocorrelation_plot(df["totOFbookings"])
     plt.title(title+ '_ACF')
     plt.grid()
     plt.show()
     # Zoom in
     n_lags = 48
     fig, ax = plt.subplots(constrained_layout=True)
     plot_acf(df["totOFbookings"], ax=ax, lags=n_lags)
     plt.title('Autocorrelation Function of '+title+'; Lags: %d' % n_lags)
     plt.grid(which='both')
     plt.xlabel("Lag")
     plt.ylabel("Autocorrelation")
     plt.show()
     #%% PACF: it gives us an initial guess on parameter P
     n_lags = 48
     fig, ax = plt.subplots(constrained_layout=True)
     plot_pacf(df["totOFbookings"], ax=ax, lags=n_lags)
     plt.title('Partial Autocorrelation Function of '+title+'; Lags: %d' % n_lags)
     plt.grid(which='both')
     plt.xlabel("Lag")
     plt.ylabel("Partial Autocorrelation")
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
                    "city": city,
                    "init_time": {"$gte": init_unix, "$lte": finish_unix}
                   }
                },
        {"$project": {
            "duration": {
                "$subtract": ["$final_time", "$init_time"]
            },

            "or_de": {
                "$ne": [{
                    "$arrayElemAt": ["$origin_destination.coordinates", 0]
                },
                    {
                        "$arrayElemAt": ["$origin_destination.coordinates", 1]
                    }
                ]
            },

            "dow": {"$dayOfYear": "$init_date"},
            "hour": {"$hour": "$init_date"}
        }},
        {"$match": {
            "duration": {
                "$gte": 5 * 60, "$lte": 150 * 60
            },
            "or_de": True
        }
        },

        {"$group": {
            "_id": {"dow": "$dow", "hour": "$hour"},
            "totOFbookings": {"$sum": 1}

        }
        },
        {"$sort": {
            "_id": 1

        }

        }])

        df = time_series_df = timeSeries_day(city + "_timeSeries", timeSeries['Bookings' + city])

        # Call the function to fill in missing combinations with zero values
        df_miss = add_miss(pd.read_csv(io.StringIO(df)),city + "_timeSeries with missed data")

        df_stat=stationarity(df_miss,city)
        df_autoc=autocorrelation(df_miss,city)
        #%% Fitting the model with initial guess #choose the order of the initial model 
        




