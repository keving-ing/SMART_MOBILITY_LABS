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
    plt.show()

    DFBookings.columns = ['_id', 'totOFbookings']

    df_result = DFBookings.to_csv(index=False)  # Save to CSV without index

    return df_result

def add_miss(df):
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

    # Sort the DataFrame by dow and hour
    df['_id'] = df['_id'].apply(lambda x: {'dow': int(x['dow']) if isinstance(x, dict) else int(x.split(',')[0].split(':')[1].strip('}{ ')),
                                       'hour': int(x['hour']) if isinstance(x, dict) else int(x.split(',')[1].split(':')[1].strip('}{ '))})
    
    df = df.sort_values(by=[('_id', 'dow'), ('_id', 'hour')]).reset_index(drop=True)


    
    plt.plot(df['rentals'])
    plt.title('Filled rentals timeseries')
    plt.xlabel('Index')
    plt.ylabel('Number of rentals')
    plt.xticks(rotation=50
               )
    plt.grid(linestyle='--', linewidth=0.8)
    plt.show()

    df.plot(y='rentals', title="Rentals in AMSTERDAM")

    return df


def dateparse(time_insex):
        return pd.datetime.fromtimestamp(float(time_insex)*3600)



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

    cities=["Amsterdam","Milano"]

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
        df_n = add_miss(pd.read_csv(io.StringIO(df)))
