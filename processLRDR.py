# -*- coding: utf-8 -*-
"""
Created on Sat May 12 22:14:42 2018

@author: pihlaart1
"""
import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from matplotlib import pyplot as plt

def loadData(test):
    if test:
        nrows = 10000
    else:
        nrows = None # Load all        
    df = pd.read_csv("..\\MilkingRobot-DR & LR files\\MilkingRobot-2.1.1-dr&DR-WithProduction-NoStrange-159k-GOOD.csv",sep=";",low_memory=False, nrows=nrows)            
    df = df.append(pd.read_csv("..\\MilkingRobot-DR & LR files\\MilkingRobot-3.3.0-LR-NoStrange-218k-GOOD (but incomplete).csv",sep=";",low_memory=False, nrows=nrows))
    return df

def normalizeFeatures(df):
    d = df.iloc[:,4:20]#.values #numpy array
    minmax_scaler = preprocessing.MinMaxScaler()
#    d_scaled = minmax_scaler.fit_transform(d) #MinMax scale
    d_scaled = preprocessing.scale(d) #Normal distribution
    d.loc[:,:] = d_scaled
    df.iloc[:,4:20] = d
    return df

def scatterPlots(df):
    plt.scatter(df["DIM"], df["Production today"])
    plt.show()

def replaceNanWithMean(col):
    pd.to_numeric(col,errors="coerce")
    avg = col.mean()
    col = col.fillna(avg)
    return col

def transformNegatives(row):
    try:
        f = float(row)
        return f
    except ValueError:
        if row[-1]=="-":
            row = "-" + row[:-1]
        f = float(row)
        return f #we'll transform later on

def setProduction(df):
#    Replace zero today with yesterday's value, if possible
    df.loc[df["Production today"]==0,"Production today"] = df.loc[df["Production today"]==0,"Production yesterday"]
    df = df[(df["Production today"]!=0) ] #remove zero production
    df.loc[:,"Production today"] = df.loc[:,"Production today"]/1000 # grams to kilograms
#    df = df.loc[df["DIM"]!=0] #should we remove zero DIM?        
    return df

def kmeansCluster(df):
    k = 4
    kmeans = cluster.KMeans(n_clusters=k)
    labels = kmeans.fit_predict(df)
    return labels

dr = loadData(test=True)
#lr = loadLR()
#dr = dr.append(lr)
#del(lr)
dr = dr[["KEY","Cow Code","Date","Farm","DIM","Calving No","Content fat percent",
        "Content protein percent","Feeding forecast","Jaanos yhteensa",
        "Lypsyja Last 24 h", #"Lyspyja avg", 
        "Ohikulut 24 h", "Production avg", "Production change",
        "Production today", "Production yesterday", "Syonti yhteensa", "Weight avg", "Yhteensa tanaan"]]

## Transform negative numbers from problematic SAP format to normal float
#for i in [13,17]:
#    dr.iloc[:,i] = dr.iloc[:,i].apply( lambda r: transformNegatives(r))

# Replace non-numeric values    
for i in range(4,19):
    dr.iloc[:,i] = replaceNanWithMean(dr.iloc[:,i])

dr = setProduction(dr)
dr = normalizeFeatures(dr)
(dr.loc[:,["DIM", "Calving No", "Production today", "Syonti yhteensa"]]).isnull().sum()
km = kmeansCluster(dr.loc[:,["DIM", "Calving No", "Production today", "Syonti yhteensa"]])

