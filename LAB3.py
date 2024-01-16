import pymongo as pm
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
import csv
import numpy as np
import re
from pandas import read_csv
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import geojson
import seaborn


def normalize_matrix(A):
    max_od = 0
    for i in A:
        m = max(i)
        max_od = m if m > max_od else max_od

    # normalization
    return [[x / max_od for x in y] for y in A]


def print_matrix(A, startHour="-", endHour="-", startDay="-", endDay="-", label="", title=""):
    np.savetxt(title + ".csv", A, delimiter=",")
    print("\n\n")
    print(f"{label} - Hours: {startHour} - {endHour}, Days: {startDay} - {endDay}")
    for r in A:
        for c in r:
            print(c, end="\t")
        print()


def print_3d(ODDistances, label=""):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=[20, 7])

    # Make data.
    X = np.arange(0, len(ODDistances), 1)
    Y = np.arange(0, len(ODDistances[0]), 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, np.array(ODDistances), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(min([min(x) for x in ODDistances]), max([max(x) for x in ODDistances]))
    ax.zaxis.set_major_locator(LinearLocator(10))

    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(label)
    plt.show()


def get_distances(A):
    norm_1 = np.linalg.norm(A, 1)  # 1-norm
    norm_2 = np.linalg.norm(A, 'fro')  # default 2-norm (frobenius-norm)
    norm_inf = np.linalg.norm(A, np.inf)
    norm_nuc = np.linalg.norm(A, 'nuc')
    return norm_1, norm_2, norm_inf, norm_nuc


if __name__ == "__main__":
    # Configuration
    client = pm.MongoClient(host='bigdatadb.polito.it',
                            port=27017,
                            ssl=True,
                            tlsAllowInvalidCertificates=True,
                            username='ictts',
                            password='Ict4SM22!',
                            authSource='carsharing',
                            authMechanism='SCRAM-SHA-1')
    db = client['carsharing']  # Choose the DB to use
    # plt.plot(range(10),range(10))
    collection = "ictts_PermanentBookings"
    ictts_PermanentBookings = db[collection].find()

    with open('zone/TorinoZonesArray.geojson') as f:
        array_zones = geojson.load(f)
    print(f"numero zone: {len(array_zones)}")
    zones = array_zones
    len_zones = len(zones)
    all_matrices_mongo = []
    
    # ALL rentals for Car2Go
    # Take the first matrix from MongoDB
    startHour = 0
    endHour = 24
    startDay = 1
    endDay = 7
    ODMatrix_1 = [[0 for y in range(len_zones)] for x in range(len_zones)]
    for i in range(23):
        # ODMatrix
        orig_zone = zones[i]
        # tot = 0
        print(f"computing for orig_zone = {i} ==> ")
        for j in range(len_zones):
            dest_zone = zones[j]
            p = db["ictts_PermanentBookings"].aggregate([
                {"$project": {
                    "hour": {"$hour": "$init_date"},
                    "day": {"$dayOfWeek": "$init_date"},
                    "init_loc": 1,
                    "final_loc": 1
                }
                },
                {"$match": {
                    "day": {
                        "$gte": startDay,
                        "$lte": endDay
                    },
                    "hour": {
                        "$gte": startHour,
                        "$lte": endHour
                    },
                    "init_loc": {
                        "$geoWithin": {
                            "$geometry": {
                                "type": "MultiPolygon",
                                "coordinates": orig_zone
                            }
                        }
                    },
                    "final_loc": {
                        "$geoWithin": {
                            "$geometry": {
                                "type": "MultiPolygon",
                                "coordinates": dest_zone
                            }
                        }
                    }
                }
                },
                {
                    "$count": "tot"
                }
            ])

            val = list(p)
            # print(val)
            if len(val) != 0:
                ODMatrix_1[i][j] = val[0]['tot']
            else:
                ODMatrix_1[i][j] = 0

            # ODMatrix[i][j] = p["tot"]
            # tot += ODMatrix[i][j]
        # ODMatrix[i][j] = tot
     #print_matrix(ODMatrix_1, startHour=startHour, endHour=endHour, startDay=startDay, endDay=endDay, label="",title="OD1")
    print("\n\n")
    print(f"OD matrix_1 - Hours: {startHour} - {endHour}, Days: {startDay} - {endDay}")
    for r in ODMatrix_1:
        for c in r:
            print(c, end="\t")
        print()
    all_matrices_mongo.append(ODMatrix_1)
    
# ALL rentals for Enjoy
    #Take the first matrix from MongoDB

    startHour = 0
    endHour = 24
    startDay = 1
    endDay = 7

    ODMatrix_1 = [[0 for y in range(len_zones)] for x in range(len_zones)]
    for i in range( len_zones):
        #ODMatrix
        orig_zone = zones[i]
        #tot = 0
        print(f"computing for orig_zone = {i} ==> ")
        for j in range (len_zones):
            dest_zone = zones[j]
            p = db["ictts_enjoy_PermanentBookings"].aggregate([
                    {"$project": {
                        "hour" : {"$hour": "$init_date"},
                        "day" : {"$dayOfWeek": "$init_date"},
                        "init_loc": 1,
                        "final_loc": 1
                        }
                    },
                    { "$match": {
                        "day": {
                                    "$gte" : startDay,
                                    "$lte": endDay
                                },
                        "hour": {
                                    "$gte" : startHour,
                                    "$lte": endHour
                                },   
                        "init_loc": {
                                "$geoWithin": {
                                        "$geometry": {
                                            "type": "MultiPolygon",
                                            "coordinates": orig_zone
                                        }
                                }
                        },
                        "final_loc": {
                                "$geoWithin": {
                                        "$geometry": {
                                            "type": "MultiPolygon",
                                            "coordinates": dest_zone
                                        }
                                }
                        }
                    }
                    },
                    {
                        "$count" : "tot"
                    }
            ])

            val = list(p)
            #print(val)
            if len(val) != 0:
                ODMatrix_1[i][j] = val[0]['tot']
            else:
                ODMatrix_1[i][j] = 0
            #ODMatrix[i][j] = p["tot"]
            #tot += ODMatrix[i][j]
        #ODMatrix[i][j] = tot
        print(ODMatrix_1[i])

    print("\n\n")
    print(f"OD matrix_1 - Hours: {startHour} - {endHour}, Days: {startDay} - {endDay}")  
    for r in ODMatrix_1:
        for c in r:
            print(c, end = "\t")
        print()

    all_matrices_mongo.append(ODMatrix_1)

# ALL rentals for Car2Go within Monday-Friday
#Take the first matrix from MongoDB
    startHour = 0
    endHour = 24
    startDay = 2
    endDay = 6
    ODMatrix_1 = [[0 for y in range(len_zones)] for x in range(len_zones)]
    for i in range( len_zones):
        #ODMatrix
        orig_zone = zones[i]
        #tot = 0
        print(f"computing for orig_zone = {i} ==> ")
        for j in range (len_zones):
            dest_zone = zones[j]
            p = db["ictts_PermanentBookings"].aggregate([
                    {"$project": {
                        "hour" : {"$hour": "$init_date"},
                        "day" : {"$dayOfWeek": "$init_date"},
                        "init_loc": 1,
                        "final_loc": 1
                        }
                    },
                    { "$match": {
                            "day": {
                                "$gte" : startDay,
                                "$lte": endDay
                                  },
                            "hour":  {
                                "$gte" : startHour,
                                "$lte": endHour
                             },
                            "init_loc": {
                                    "$geoWithin": {
                                          "$geometry": {
                                               "type": "MultiPolygon",
                                               "coordinates": orig_zone
                                               }
                                            } 
                                        },
                            "final_loc": {
                                    "$geoWithin": {
                                            "$geometry": {
                                                "type": "MultiPolygon",
                                                "coordinates": dest_zone
                                            }
                                    }
                            }
                    }
                    },
                    {
                        "$count": "tot"
                    }
            ])

            val=list(p)
            #print(val)
            if len(val) != 0:
                ODMatrix_1[i][j] = val[0]['tot']
            else:
                ODMatrix_1[i][j] = 0

            #ODMatrix[i][j] = p["tot"]
            #tot += ODMatrix[i][j]
        #ODMatrix[i][j] = tot
        print(ODMatrix_1[i])

    print("\n\n")
    print(f"OD matrix_1 - Hours: {startHour} - {endHour}, Days: {startDay} - {endDay}")
    for r in ODMatrix_1:
        for c in r:
            print(c, end = "\t")
        print()
    all_matrices_mongo.append(ODMatrix_1)    

# ALL rentals for Enjoy within Saturday-Sunday
    #Take the first matrix from MongoDB

    startHour = 0
    endHour = 24
    startDay = 1
    endDay = 7

    ODMatrix_1 = [[0 for y in range(len_zones)] for x in range(len_zones)]
    for i in range( len_zones):
        #ODMatrix
        orig_zone = zones[i]
        #tot = 0
        print(f"computing for orig_zone = {i} ==> ")
        for j in range (len_zones):
            dest_zone = zones[j]
            p = db["ictts_enjoy_PermanentBookings"].aggregate([
                    {"$project": {
                        "hour" : {"$hour": "$init_date"},
                        "day" : {"$dayOfWeek": "$init_date"},
                        "init_loc": 1,
                        "final_loc": 1
                        }
                    },
                    { "$match": {
                        "day": {
                                    "$in" :[startDay,endDay] 
                                },
                        "hour": {
                                    "$gte" : startHour,
                                    "$lte": endHour
                                },   
                        "init_loc": {
                                "$geoWithin": {
                                        "$geometry": {
                                            "type": "MultiPolygon",
                                            "coordinates": orig_zone
                                        }
                                }
                        },
                        "final_loc": {
                                "$geoWithin": {
                                        "$geometry": {
                                            "type": "MultiPolygon",
                                            "coordinates": dest_zone
                                        }
                                }
                        }
                    }
                    },
                    {
                        "$count" : "tot"
                    }
            ])

            val = list(p)
            #print(val)
            if len(val) != 0:
                ODMatrix_1[i][j] = val[0]['tot']
            else:
                ODMatrix_1[i][j] = 0
            #ODMatrix[i][j] = p["tot"]
            #tot += ODMatrix[i][j]
        #ODMatrix[i][j] = tot
        print(ODMatrix_1[i])

    print("\n\n")
    print(f"OD matrix_1 - Hours: {startHour} - {endHour}, Days: {startDay} - {endDay}")  
    for r in ODMatrix_1:
        for c in r:
            print(c, end = "\t")
        print()

    all_matrices_mongo.append(ODMatrix_1)

# ALL rentals for Car2Go ,7-10h, within Monday-Friday
#Take the first matrix from MongoDB
    startHour = 7
    endHour = 10
    startDay = 2
    endDay = 6
    ODMatrix_1 = [[0 for y in range(len_zones)] for x in range(len_zones)]
    for i in range( len_zones):
        #ODMatrix
        orig_zone = zones[i]
        #tot = 0
        print(f"computing for orig_zone = {i} ==> ")
        for j in range (len_zones):
            dest_zone = zones[j]
            p = db["ictts_PermanentBookings"].aggregate([
                    {"$project": {
                        "hour" : {"$hour": "$init_date"},
                        "day" : {"$dayOfWeek": "$init_date"},
                        "init_loc": 1,
                        "final_loc": 1
                        }
                    },
                    { "$match": {
                            "day": {
                                "$gte" : startDay,
                                "$lte": endDay
                                  },
                            "hour":  {
                                "$gte" : startHour,
                                "$lte": endHour
                             },
                            "init_loc": {
                                    "$geoWithin": {
                                          "$geometry": {
                                               "type": "MultiPolygon",
                                               "coordinates": orig_zone
                                               }
                                            } 
                                        },
                            "final_loc": {
                                    "$geoWithin": {
                                            "$geometry": {
                                                "type": "MultiPolygon",
                                                "coordinates": dest_zone
                                            }
                                    }
                            }
                    }
                    },
                    {
                        "$count": "tot"
                    }
            ])

            val=list(p)
            #print(val)
            if len(val) != 0:
                ODMatrix_1[i][j] = val[0]['tot']
            else:
                ODMatrix_1[i][j] = 0

            #ODMatrix[i][j] = p["tot"]
            #tot += ODMatrix[i][j]
        #ODMatrix[i][j] = tot
        print(ODMatrix_1[i])

    print("\n\n")
    print(f"OD matrix_1 - Hours: {startHour} - {endHour}, Days: {startDay} - {endDay}")
    for r in ODMatrix_1:
        for c in r:
            print(c, end = "\t")
        print()
    all_matrices_mongo.append(ODMatrix_1)    

# ALL rentals for Car2Go 16-19h, within Saturday-Sunday
#Take the first matrix from MongoDB
    startHour = 16
    endHour = 19
    startDay = 1
    endDay = 7
    ODMatrix_1 = [[0 for y in range(len_zones)] for x in range(len_zones)]
    for i in range( len_zones):
        #ODMatrix
        orig_zone = zones[i]
        #tot = 0
        print(f"computing for orig_zone = {i} ==> ")
        for j in range (len_zones):
            dest_zone = zones[j]
            p = db["ictts_PermanentBookings"].aggregate([
                    {"$project": {
                        "hour" : {"$hour": "$init_date"},
                        "day" : {"$dayOfWeek": "$init_date"},
                        "init_loc": 1,
                        "final_loc": 1
                        }
                    },
                    { "$match": {
                            "day": {
                                "$in" : [startDay,endDay]
                                  },
                            "hour":  {
                                "$gte" : startHour,
                                "$lte": endHour
                             },
                            "init_loc": {
                                    "$geoWithin": {
                                          "$geometry": {
                                               "type": "MultiPolygon",
                                               "coordinates": orig_zone
                                               }
                                            } 
                                        },
                            "final_loc": {
                                    "$geoWithin": {
                                            "$geometry": {
                                                "type": "MultiPolygon",
                                                "coordinates": dest_zone
                                            }
                                    }
                            }
                    }
                    },
                    {
                        "$count": "tot"
                    }
            ])

            val=list(p)
            #print(val)
            if len(val) != 0:
                ODMatrix_1[i][j] = val[0]['tot']
            else:
                ODMatrix_1[i][j] = 0

            #ODMatrix[i][j] = p["tot"]
            #tot += ODMatrix[i][j]
        #ODMatrix[i][j] = tot
        print(ODMatrix_1[i])

    print("\n\n")
    print(f"OD matrix_1 - Hours: {startHour} - {endHour}, Days: {startDay} - {endDay}")
    for r in ODMatrix_1:
        for c in r:
            print(c, end = "\t")
        print()
    all_matrices_mongo.append(ODMatrix_1)
      
    IMQ_matrices_path = ["OD1.csv","OD2.csv","OD3.csv","OD4.csv","OD5.csv","OD6.csv","ODTOT.csv"]  
    IMQ_matrices = []

    for p in IMQ_matrices_path:
        df = read_csv(p,delimiter=",")
        data = df.fillna(0).values
        print(f"{p} -> Matrix correctly uploaded")
        IMQ_matrices.append(data)
    print("IMQ matrices caricate correttamente")
    print(len(IMQ_matrices))
    # normalization
    all_matrices_mongo_normalized = copy.deepcopy(all_matrices_mongo)
    IMQ_matrices_normalized = copy.deepcopy(IMQ_matrices)
    for i, m in enumerate(all_matrices_mongo):
        print(m)
        all_matrices_mongo_normalized[i] = m / np.linalg.norm(m, axis=1)

    for j, imq in enumerate(IMQ_matrices):
        # print("IMQ: " + imq.dtype())
        IMQ_matrices_normalized[j] = imq / np.linalg.norm(imq, axis=1, keepdims=True)
        print(f"Normalized Matrix {j}:\n{IMQ_matrices_normalized[j]}")

    print(all_matrices_mongo_normalized)

    norm_1_all = []
    norm_2_all = []
    norm_inf_all = []
    norm_nuc_all = []

    odr = []
    odimq = []
    for i, m in enumerate(all_matrices_mongo_normalized):
        for j, imq in enumerate(IMQ_matrices_normalized):
            dist = m - imq
            norm_1, norm_2, norm_inf, norm_nuc = get_distances(dist)
            odr.append(i + 1)
            odimq.append(j + 7)
            norm_1_all.append(norm_1)
            norm_2_all.append(norm_2)
            norm_inf_all.append(norm_inf)
            norm_nuc_all.append(norm_nuc)

            print(f"Distance between Carsharing [{i + 1}] and IMQ dataset [{j + 7}]:")
            print(f"Norm 1: {norm_1}, Norm 2: {norm_2}, Norm Inf: {norm_inf}, Norm Nuclear: {norm_nuc}")

    d_matrix_tmp = all_matrices_mongo_normalized[0] - IMQ_matrices_normalized[0]
    print_3d(d_matrix_tmp, label=f"Distance Matrix Carsharing[1]-IMQdataset[7]")

    d_matrix_tmp = all_matrices_mongo_normalized[0] - IMQ_matrices_normalized[3]
    print_3d(d_matrix_tmp, label=f"Distance Matrix Carsharing[1]-IMQdataset[10]")

    for i, p in enumerate(all_matrices_mongo_normalized):
        print_3d(p, label=f"Carsharing ")
    for i, p in enumerate(IMQ_matrices_normalized):
        print_3d(p, label=f"IMQ dataset ")

    

    #print(len(norm_1_all))
    results = {"CarSharing":odr, "IMQ dataset":odimq, "dist1": norm_1_all,"dist2": norm_2_all, "dist3": norm_inf_all, "dist4": norm_nuc_all}
    #dist = [11, 12, 13, 14, 15, 16, 21, 22, 23]

    results = pd.DataFrame(results)
    fig=plt.figure(figsize = [6,6])
    heat_df_mpe = results.pivot(index='CarSharing', columns='IMQ dataset',values='dist1')
    ax=seaborn.heatmap(heat_df_mpe, annot=True, linewidths = .5, fmt='.3f')
    bottom,top=ax.get_ylim()
    ax.set_ylim(bottom+0.5,top-0.5)
    plt.title('1-norm heatmap car2go')
    plt.show()
    results = pd.DataFrame(results)
    fig=plt.figure(figsize = [6,6])
    heat_df_mpe = results.pivot(index='CarSharing', columns='IMQ dataset',values='dist2')
    ax=seaborn.heatmap(heat_df_mpe, annot=True, linewidths = .5, fmt='.3f') 
    bottom,top=ax.get_ylim()
    ax.set_ylim(bottom+0.5,top-0.5)
    plt.title('1-norm heatmap car2go')
    plt.show()

    results = pd.DataFrame(results)
    fig=plt.figure(figsize = [6,6])
    heat_df_mpe = results.pivot(index='CarSharing', columns='IMQ dataset',values='dist3')
    ax=seaborn.heatmap(heat_df_mpe, annot=True, linewidths = .5, fmt='.3f')
    bottom,top=ax.get_ylim()
    ax.set_ylim(bottom+0.5,top-0.5)
    plt.title('1-norm heatmap car2go')
    plt.show()
    results = pd.DataFrame(results)
    fig=plt.figure(figsize = [6,6])
    heat_df_mpe = results.pivot(index='CarSharing', columns='IMQ dataset',values='dist4')
    ax=seaborn.heatmap(heat_df_mpe, annot=True, linewidths = .5, fmt='.3f')
    bottom,top=ax.get_ylim()
    ax.set_ylim(bottom+0.5,top-0.5)
    plt.title('1-norm heatmap car2go')
    plt.show()

    print("siamo alla fine")