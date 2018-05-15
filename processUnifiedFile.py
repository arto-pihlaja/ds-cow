# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:39:14 2018

@author: pihlaart1
"""
import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from matplotlib import pyplot as plt
from processLRDR import transformNegatives
from confidential import confidential #Value calculating formulas, not shared


def loadData(test):
    if test:
        nrows = 10000
    else:
        nrows = None # Load all        
    df = pd.read_csv("..\\UnifiedMilkingRobot-1.0-LRDRMemory-350k\\AUnifiedMilkingRobot-1.0-LRDRMemory-350k.csv",sep=";",decimal=",", low_memory=False, nrows=nrows, encoding='latin1')            
    df.loc[:,"Date"] = pd.to_datetime(arg=df.loc[:,"Date"],format="%Y%m%d") 
    df.loc[:,"Max Analysis date"] = pd.to_datetime(arg=df.loc[:,"Max Analysis date"],format="%Y%m%d",errors="coerce") 
    df.loc[:,"Analyis Date"] = pd.to_datetime(arg=df.loc[:,"Analyis Date"],format="%Y%m%d" ,errors="coerce") 
    df.sort_values(by="Date",inplace=True)    
    df["Content protein perc"] = [x.replace(",",".") for x in df["Content protein perc"]]
    df["Content protein perc"] = df["Content protein perc"].apply( lambda r: transformNegatives(r))
    df["Content protein perc"] = df["Content protein perc"].astype(float)
    return df

def discardAnomalies(df): #drops about 80,000 rows
    dateLimiter = (df.loc[:,"Date"] > "2017-1-1")    
    df = df.loc[dateLimiter]
    dimLimiter = (df.loc[:,"DIM"] < 600)    
    df = df.loc[dimLimiter]
    passLimiter = (df.loc[:,"Ohikulut 24 h"] < 10)
    df = df.loc[passLimiter]
    feedLimiter = (df.loc[:,"Syonti yhteensa"] < 20)
    df = df.loc[feedLimiter]    
#   cut PrevLypsyjä and Lypsyjä by 10
#    cut Prevsyönti by 20
#    cut PrevDIM by 600
#    cut PrevCalving by 12
#    weightAvgLimiter = (df.loc[:,"Weight avg"] < 900)
#    df = df.loc[weightAvgLimiter]        
#    productionAvgLimiter = (df.loc[:,"Production avg"] < 80)
#    df = df.loc[productionAvgLimiter]            
#    productionTodayLimiter = (df.loc[:,"Production today"] < 80)
#    df = df.loc[productionTodayLimiter]                
    fatLimiter = (df.loc[:,"Content fat percent"] < 0.06)
    df = df.loc[fatLimiter]        
    return df

def buildFarmAverages(df):
    aggregation = {"DIM":"mean",
                   "Production today":"mean",                   
                   "Production avg":"mean",
                   "Calving No":"mean",
                   "Lypsyja Last 24 h":"mean",
                   "Ohikulut 24 h":"mean",
                   "Syonti yhteensa":"mean",
                   "Cow Code":"count"}
    farmAvgDf = df.groupby(["Farm","Date","Records source"], as_index=False).agg(aggregation)
    return farmAvgDf    

#def get51663():    
#    Feeding plan integration


#%% cell - help deciding which percentage to use
def countPercentages(df):
    print("Number of non-zero Content fat percent: ", (df["Content fat percent"]!=0).sum() )
    print("Number of non-zero Last Content Fat Percent: ", (df["Last Content Fat Percent"]!=0).sum() )
    print("Number of non-zero Content protein perc: ", (df["Content protein perc"]!=0).sum() )
    print("Number of non-zero Last Content Protein Percent: ", (df["Last Content Protein Percent"]!=0).sum() )    
#%% cell - clustering     

def normalizeFeatures(df):
#    minmax_scaler = preprocessing.MinMaxScaler()
#    d_scaled = minmax_scaler.fit_transform(d) #MinMax scale
    d_scaled = preprocessing.scale(df) #Normal distribution
    df.loc[:,:] = d_scaled
    return df

def clusterCows(df, k):
    kmeans = cluster.KMeans(n_clusters=k,init="k-means++") # parallelization with n_jobs=-2 caused issues
    c = kmeans.fit(df)
    return c

def kElbow(df,kstart,kstop):
    kvalue = []
    cluster = []
    plt.figure(2)
    inertia = []
#    cluster by calculated value, calving no, DIM, lypsyjä, syönti yht
    df = df.loc[:,["CalcValue", "Calving No", "DIM", "Lypsyja Last 24 h","Syonti yhteensa"]]
    df = normalizeFeatures(df)    
    for k in range(kstart,kstop):
        kvalue.append(k)
        print("Calculating for k = ",k)
        cowCluster = clusterCows(df, k)
        cluster.append(cowCluster)        
        print("Inertia for k = ", k, " is ", cowCluster.inertia_)
        if k > 3:
            i = k - kstart - 1
            print("Inertia dropped by ", inertia[i] - cowCluster.inertia_ )
        inertia.append(cowCluster.inertia_)        
    clusterDf = pd.DataFrame({
                "kvalue": kvalue,
                "cluster": cluster,
                "inertia": inertia
            })
    plt.plot(kvalue,inertia)
    plt.show()
    input("Enter to continue")
    return(clusterDf)
    
def calculateClusterStats(df):
    aggregation = {"DIM":"mean",
                   "Production today":"mean",                   
                   "Production avg":"mean",
                   "Calving No":"mean",
                   "Lypsyja Last 24 h":"mean",
                   "Ohikulut 24 h":"mean",
                   "Syonti yhteensa":"mean",
                   "Cow Code":"count"}
    clusterAvgDf = df.groupby(["CowCluster","Farm","Date","Records source"], as_index=False).agg(aggregation)
    return clusterAvgDf


#%% - MAIN
    
df = loadData(test=False)
print("df initially has ",df.shape[0], " rows")
df = discardAnomalies(df)
print("After cleansing we have ",df.shape[0], " rows")
stats =df.describe(exclude=["object"])
farmAvgDf = buildFarmAverages(df)
farmAvgDf.to_csv(".\\data\\farmAverages.csv",sep=";",decimal=",")
print("Saved farm averages to farmAverages.csv")
df = confidential.calculateValue(df)
cdf = kElbow(df, 3, 12)
istr = input("Enter number of clusters to use: ")
idx = int(istr) - 3
c = cdf.loc[idx,"cluster"]
df["CowCluster"] = c.labels_
plt.figure(3)
plt.scatter(c.cluster_centers_[:,4],c.cluster_centers_[:,0])
plt.title("KMeans cluster centroids, x = Feed total, y = Calculated value")
plt.show()    


#df["CowCluster"] = cowCluster.labels_

