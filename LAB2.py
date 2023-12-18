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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score)




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

    # plt.figure()
    # plt.xlabel("Days of October")
    # plt.ylabel("N. of Bookings")
    # plt.title(title)
    # plt.plot(DFBookings["totOFbookings"], label="Bookings")
    # plt.xticks(ticks=ticks,
    #            labels=labels,
    #            rotation=-30)
    # plt.legend(loc='best')
    # plt.grid(True, which="both")

    # plt.show()

    #plt.show()


    DFBookings.columns = ['_id', 'totOFbookings']

    df_result = DFBookings.to_csv(index=False)  # Save to CSV without index

    return df_result


def add_miss(df, title):
    dow_hour_combinations = [(dow, hour) for dow in range(275, 305) for hour in range(24)]
    new_rows = []

    print(df)
    for dow, hour in dow_hour_combinations:
        # Check if the combination exists in the DataFrame
        if not any(((eval(entry['_id'])['dow'] == dow) and (eval(entry['_id'])['hour'] == hour)) for _, entry in
                   df.iterrows()):
            # If not, create a new row with rentals value set to 0
            new_rows.append({'_id': {'dow': dow, 'hour': hour}, 'totOFbookings': 0})

    # Create a new DataFrame with the new rows
    new_df = pd.DataFrame(new_rows)

    # Concatenate the original DataFrame and the new DataFrame
    df = pd.concat([df, new_df], ignore_index=True)

    print(df)

    df[['dow', 'hour']] = df['_id'].apply(
        lambda x: pd.Series([x['dow'], x['hour']] if isinstance(x, dict) else x.split(',')))

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
    # plt.figure()
    # plt.xlabel("Days of October")
    # plt.ylabel("N. of Bookings")
    # plt.title(title)
    # plt.plot(df["totOFbookings"], label="Bookings")
    # plt.xticks(ticks=ticks,
    #            labels=labels,
    #            rotation=-30)
    # plt.legend(loc='best')
    # plt.grid(True, which="both")


    
#NON SCOMMENTARE ERA GIA COMMENTATO***
    # plt.show()

   # plt.show()


    #     plt.plot(df['totOFbookings'])
    #     plt.title('Filled rentals timeseries')
    #     plt.xlabel('Index')
    #     plt.ylabel('Number of rentals')
    #     plt.xticks(rotation=50
    #                )
    #     plt.grid(linestyle='--', linewidth=0.8)
    #     plt.show()

    # df.plot(y='totOFbookings', title="Rentals in AMSTERDAM")
#NON SCOMMENTARE ERA GIA COMMENTATO***
    return df



def stationarity(df, title):
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

    df['MA'] = df['totOFbookings'].rolling(24 * 7).mean()  # Moving average
    df['MS'] = df['totOFbookings'].rolling(24 * 7).std()  # Moving std
    # plt.figure(constrained_layout=True)
    # plt.plot(df['totOFbookings'], linewidth=1, label='Number of rentals')
    # plt.plot(df['MA'], linewidth=2, color='r', label='Moving Average')
    # plt.plot(df['MS'], linewidth=2, label='Moving Std')
    # plt.title('Moving Average/Std of bookings of ' + title)
    # plt.xlabel('Date')
    # plt.ylabel('Number of bookings')
    # plt.xticks(ticks=ticks,
    #            labels=labels,
    #            rotation=-30)
    # plt.grid(linestyle='--', linewidth=0.8)
    # plt.legend()
    # plt.show()

    return df


def autocorrelation(df, title):
    # plt.figure(constrained_layout=True)
    # pd.plotting.autocorrelation_plot(df["totOFbookings"])
    # plt.title(title + '_ACF')
    # plt.grid()
    # plt.show()
    # Zoom in
    n_lags = 48
    fig, ax = plt.subplots(constrained_layout=True)
    # plot_acf(df["totOFbookings"], ax=ax, lags=n_lags)
    # plt.title('Autocorrelation Function of ' + title + '; Lags: %d' % n_lags)
    # plt.grid(which='both')
    # plt.xlabel("Lag")
    # plt.ylabel("Autocorrelation")
    # plt.show()
    # %% PACF: it gives us an initial guess on parameter P
    n_lags = 48
    fig, ax = plt.subplots(constrained_layout=True)
    # plot_pacf(df["totOFbookings"], ax=ax, lags=n_lags)
    # plt.title('Partial Autocorrelation Function of ' + title + '; Lags: %d' % n_lags)
    # plt.grid(which='both')
    # plt.xlabel("Lag")
    # plt.ylabel("Partial Autocorrelation")
    # plt.show()

def split(df):
    data = df["totOFbookings"].values.astype(float)
    N = 7 * 24
    train_set, test_set = data[0:N], data[N:(2 * N)]

    return train_set, test_set, data


##TASK 6
def model_training(train, test, data, title):
    order = (2, 0, 2)  # p = 2 , q = 2
    model = ARIMA(train.astype(float), order=order)
    model_fit = model.fit(method='statespace')

    # print(model_fit.summary())

    # plt.plot(train, label='Original')
    # plt.plot(model_fit.fittedvalues, color="red", label='Forecasted')
    # plt.title('Original/Forecast timeseries of ' + title + ' - Training phase')
    # plt.xlabel("Date")
    # plt.ylabel("Number of rentals")
    # plt.xticks(rotation=50)
    # plt.legend(loc='upper right')
    # plt.grid(linestyle='--', linewidth=0.8)
    # plt.show()

    ##ERROR METRICS

    mae = mean_absolute_error(train[0:len(model_fit.fittedvalues)], model_fit.fittedvalues)
    mape = mae / np.mean(train[0:len(model_fit.fittedvalues)]) * 100
    print("TRAIN DATASET : (%i,0,%i) model => MAE: %.3f -- MSE: %.3f -- R2: %.3f -- MAPE: %.3f" % (2, 2,
                                                                                                   mean_absolute_error(
                                                                                                       train[0:len(
                                                                                                           model_fit.fittedvalues)],
                                                                                                       model_fit.fittedvalues),
                                                                                                   mean_squared_error(
                                                                                                       train[0:len(
                                                                                                           model_fit.fittedvalues)],
                                                                                                       model_fit.fittedvalues),
                                                                                                   r2_score(train[0:len(
                                                                                                       model_fit.fittedvalues)],
                                                                                                            model_fit.fittedvalues),
                                                                                                   mape))

    ##MODEL TESTING AND ERROR METRICS

    history = train.astype(float)
    predictions = []
    for t in range(0, len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit(method='statespace')
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history = np.append(history, obs)  # expanding window
    # plots
    # plt.plot(test, label='Original')
    # plt.plot(predictions, color="red", label='Forecasted')
    # plt.title('Original/Forecast timeseries of ' + title + ' - Testing phase')
    # plt.xlabel("Date")
    # plt.ylabel("Number of rentals")
    # plt.xticks(rotation=50)
    # plt.legend(loc='best')
    # plt.grid(linestyle='--', linewidth=0.8)
    # plt.show()

    mae = mean_absolute_error(test, predictions)
    mape = mae / np.mean(test[0:len(model_fit.fittedvalues)]) * 100

    print(str(mae) + "      " + str(mape))
    # %% Fitted initial model
    # plt.plot(data[0:len(model_fit.fittedvalues)], label='Original')
    # plt.plot(model_fit.fittedvalues, color="red", label='Forecasted')
    # plt.title('Original/Forecast timeseries of ' + title)
    # plt.xlabel("Date")
    # plt.ylabel("Number of rentals")
    # plt.xticks(rotation=50)
    # plt.legend(loc='upper right')
    # plt.grid(linestyle='--', linewidth=0.8)

    # residuals = pd.DataFrame(model_fit.resid)
    # residuals.plot()
    # residuals.plot(kind='kde')
    # plt.title('Model Residuals_ARIMA ' + title)
    # plt.xlabel("Residual")
    # plt.ylabel("Density")
    # plt.show()
# TASK7
def variation(df,title,q,d):

    data=df['totOFbookings'].values.astype(float)
    train,test= data[0:7*24], data [7*24:(2*7*24)]
    len_test=len(test)
    p_var=(1,2,3)

    predictions = np.zeros((len(p_var),len_test))
    results = {"p": [], "d": [], "q": [], "mse": [], "mae": [], "mape": []}
    try:
        for p in p_var:
            print('Testing ARIMA order (%i, 0, %i)' % (p,q))
            train, test = data[0:7*24], data[7*24:(7*24+len_test)]
            history = [x for x in train]
            for t in range(0, len_test):
                model = ARIMA(history, order= (p, d, q))
                model_fit = model.fit( method='statespace')
                output = model_fit.forecast()
                yhat = output[0]
                predictions[p_var.index(p)][t] = yhat
                obs = test[t]
                history.append(obs) #expanding window
                history=history[1:]
    except Exception as e:
        print(f"Si è verificata un'eccezione di tipo {type(e).__name__}: {str(e)}")
        pass

    plt.plot(test,color = 'black', label = "Original")
    for p in p_var:
        print("(%i,0,2) model => MAE: %.3f -- MSE: %.3f -- R2: %.3f" %(p,
                                                                       mean_absolute_error(test,predictions[p_var.index(p)]),
                                                                       mean_squared_error(test,predictions[p_var.index(p)]),
                                                                       r2_score(test,predictions[p_var.index(p)])))


        mae = mean_absolute_error(test,predictions[p_var.index(p)])
        mape = mae/np.mean(test)*100
        results["p"].append(p)
        results["d"].append(0)
        results["q"].append(2)
        results["mse"].append(mean_squared_error(test,predictions[p_var.index(p)]))
        results["mae"].append(mean_absolute_error(test,predictions[p_var.index(p)]))
        results["mape"].append(mape)
        plt.plot(predictions[p_var.index(p)],label='p=%i' %p)

    plt.title('Parameter p variation for '+ city)
    plt.xlabel(" hours")
    plt.ylabel("Number of rentals")
    plt.legend(loc='best')
    plt.grid(linestyle = '--', linewidth=0.8)
    plt.show()

 # %% Parameter q variation
    # testing

    p = 2  # insert the p with lowest error here
    MA_orders = (1, 2, 3)
    predictions = np.zeros((len(p_var), len_test))
    for q in MA_orders:
        print('Testing ARIMA order (%i, 0, %i)' % (p, q))
        train, test = data[0:7*24], data[7*24:(7*24 + len_test)]
        history = [w for w in train]
        for t in range(0, len_test):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit(method='statespace')
            output = model_fit.forecast()
            yhat = output[0]
            predictions[p_var.index(q)][t] = yhat
            obs = test[t]
            history.append(obs)  # expanding window
    # plotting and metrics
    plt.plot(test, color="black", label="Original")
    for q in MA_orders:
        print("(%i,0,%i) model => MAE: %.3f -- MSE: %.3f -- R2: %.3f" % (p, q,mean_absolute_error(test, predictions[p_var.index(q)])
        , mean_squared_error(test, predictions[p_var.index(q)])
        , r2_score(test, predictions[p_var.index(q)])))
        mae = mean_absolute_error(test, predictions[p_var.index(p)])
        mape = mae / np.mean(test) * 100
        results["p"].append(p)
        results["d"].append(0)
        results["q"].append(q)
        results["mse"].append(mean_squared_error(test, predictions[p_var.index(q)]))
        results["mae"].append(mean_absolute_error(test, predictions[p_var.index(q)]))
        results["mape"].append(mape)
        plt.plot(predictions[p_var.index(q)], label='q=%i' % q)

    plt.title('Parameter q variation for '+ city)
    plt.xlabel(" hours")
    plt.ylabel("Number of rentals")
    plt.legend(loc='best')
    plt.grid(linestyle = '--', linewidth=0.8)
    plt.show()

def variation_p_d(df):


    data = df['totOFbookings'].values.astype(float)
    train, test = data[0:7 * 24], data[7 * 24:(2 * 7 * 24)]
    test_len = len(test)

    N = 7 * 24  # amount of data for training train, test = data[0:N], data[N:(2*N)] test_len = len(test)
    lag_orders = (1, 2, 3, 4, 5)
    MA_orders = (1, 2, 3, 4, 5)
    # predictions = np.zeros((len(lag_orders), test_len))
    predictions = np.zeros(((len(lag_orders) * len(MA_orders)), test_len))
    results = {"p": [], "d": [], "q": [], "mse": [], "mae": [], "mape": [], "mpe": []}
    combinations = range(0, (len(lag_orders) * len(MA_orders)))


    comb = 0
    for p in lag_orders:
        for q in MA_orders:
            print('Testing ARIMA order (%i, %i, %i)' % (p, 0, q))
            train, test = data[0:N], data[N:(N + test_len)]
            history = [x for x in train]
            try:
                for t in range(0, test_len):
                    model = ARIMA(history, order=(p, 0, q))
                    model_fit = model.fit( method='statespace')
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions[comb][t] = yhat
                    obs = test[t]
                    history.append(obs)  # expanding window #to make sliding window
    # history = history[1:]
                print("(%i,%i,%i) model => MAE: %.3f -- MSE: %.3f -- R2: %.3f" % (p, 0, q,
                                                                                  mean_absolute_error(test, predictions[comb]),
                mean_squared_error(test, predictions[comb]), r2_score(test, predictions[comb])))
                mae = mean_absolute_error(test, predictions[comb])
                mape = mae / np.mean(test) * 100
                adder = [(a - b) / a for a, b in zip(test, predictions[comb])]
                mpe = (100 / test_len) * np.sum(adder)
                results["p"].append(p)
                results["d"].append(d)
                results["q"].append(q)
                results["mse"].append(mean_squared_error(test, predictions[comb]))
                results["mae"].append(mean_absolute_error(test, predictions[comb]))
                results["mape"].append(mape)
                results["mpe"].append(mpe)
                comb += 1
            except:
                pass
            







     

     


if __name__ == "__main__":

    client = pm.MongoClient('bigdatadb.polito.it',
                            port=27017,
                            ssl=True,
                            tlsAllowInvalidCertificates=True,
                            username='ictts',
                            password='Ict4SM22!',
                            authSource='carsharing',
                            # authMechanism='SCRAM-SHA-1'
                            )
    db = client['carsharing']
    Bookings = db['PermanentBookings']

    cities = ["Denver", "Amsterdam", "Milano"]

    # ---------------------------------------------------------------------------------------------

    # STEP_3 - TASK1
    # Setting of the period of time
    date_init = datetime.strptime('2017-10-01T22:00:00', '%Y-%m-%dT%H:%M:%S')
    date_finish = datetime.strptime('2017-10-31T23:59:59', '%Y-%m-%dT%H:%M:%S')
    # date_init_Denver = datetime.strptime('2017-10-01T06:00:00','%Y-%m-%dT%H:%M:%S')
    # date_finish_Denver = datetime.strptime('2017-10-30T07:59:59','%Y-%m-%dT%H:%M:%S')
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

        df_miss = add_miss(pd.read_csv(io.StringIO(df)), city + "_timeSeries with missed data")

        df_stat = stationarity(df_miss, city)
        df_autoc = autocorrelation(df_miss, city)
        # %% Fitting the model with initial guess #choose the order of the initial model

        split(df_miss)

        train, test, data = split(df_miss)

        model_training(train, test, data, city)
        #variation(df_miss,"variation over coefficient p",2,0)
        variation_p_d(df_miss)
        