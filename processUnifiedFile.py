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

#%% data loading
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
    if df.dtypes["Content protein perc"] == object:
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
    pcLimiter = (df.loc[:,"PrevCalving"] < 11)
    df = df.loc[pcLimiter]        
    pdLimiter = (df.loc[:,"PrevDIM"] < 600)
    df = df.loc[pdLimiter]        
    pmLimiter = (df.loc[:,"PrevLypsyja"] < 10)
    df = df.loc[pmLimiter]        
    pfLimiter = (df.loc[:,"PrevSyonti"] < 20)
#    weightAvgLimiter = (df.loc[:,"Weight avg"] < 900)
#    df = df.loc[weightAvgLimiter]        
#    productionAvgLimiter = (df.loc[:,"Production avg"] < 80)
#    df = df.loc[productionAvgLimiter]            
#    productionTodayLimiter = (df.loc[:,"Production today"] < 80)
#    df = df.loc[productionTodayLimiter]                
    fatLimiter = (df.loc[:,"Content fat percent"] < 0.06)
    df = df.loc[fatLimiter]        
    return df

def loadAndCleanse(test=False):
    df = loadData(test)
    print("df initially has ",df.shape[0], " rows")
    df = discardAnomalies(df)
    print("After cleansing we have ",df.shape[0], " rows")
    return df
    
#%% farm Averages
    
def buildFarmAverages(df):
    aggregation = {"DIM":["mean","std"],
                   "Production today":["mean","std","sem","mad","min","max"] ,                  
                   "Production avg":["mean","std"],
                   "Calving No":["mean","std"],
                   "Lypsyja Last 24 h":["mean","std"],
                   "Ohikulut 24 h":["mean","std"],
                   "Syonti yhteensa":["mean","std","min","max"],
                   "Cow Code":"count"}
#    also calculate mean and std for protein and fat content. For 51663 use content fat, not last content
    farmGb = df.groupby(["Farm","Date","Records source"],as_index=False)
    farmAgg = farmGb.agg(aggregation)
#    farmAgg["DIM20"] = (farmGb["DIM"]<20).sum()
#    farmAvgDf.to_csv(".\\data\\farmAverages.csv",sep=";",decimal=",")
#    print("Saved farm averages to farmAverages.csv")
    return farmAgg    

def buildFarmLevelData(df):
    df = buildFarmAverages(df)
    

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
        if k > kstart:
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
    clusterAvgDf = df.groupby(["Farm","Date","CowCluster","Records source"], as_index=False).agg(aggregation)
# pick just a few dates for clarity    
    return clusterAvgDf

def doClustering(df):
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
    
#%% - Feeding plan integration
            
class FeedingPlan:    
    def __init__(self,df,farm):
        self.df = df
        self.farm = farm
        fp = pd.read_csv("..\\Tuotantotutka_files\\TUOTOS_FEEDING_PLAN_20180508_134717.csv",sep=";")
        columnNames = ["Date", "FeedType", "FeedName", "Farm", "X4", "X5", "X6",
                       "F10", "F15", "F20", "F25", "F30", "F35", "F40", "F45", "F50", "F55", "F60"] 
        fp.columns = columnNames            
        fp = fp.loc[(fp["FeedType"]=="Aine, kiloja") & (fp["Farm"]== farm),:]
        fp.loc[:,"Date"] = pd.to_datetime(arg=fp.loc[:,"Date"],format="%Y%m%d")
        self.fp = fp
    
    def nearest(items, pivot):
        return min(items, key= lambda x: abs(x - pivot))
    
    def getFeedForRow(r, self, row, feed):
        nearestFeedPlanDate = FeedingPlan.nearest(self.fp.loc[:,"Date"], row["Date"])
        feedPlanRow = self.fp.loc[( (self.fp["Date"]==nearestFeedPlanDate) & (self.fp["FeedName"]==feed) ),:]
        fr = feedPlanRow["F20"]
        return fr.iloc[0]

    def integrateFeedPlan(self):    #    Feeding plan integration to farm level daily aggregates
        # limit the full cowdataset to farm dataset
        self.df = self.df.loc[(self.df.loc[:,"Farm"]==self.farm),:]    
        print("Total ", self.df.shape[0], " rows for farm ", self.farm)        
        farmAvg = buildFarmAverages(self.df)
#        query = " 'Farm = ", farm, "' "
#        df = df.query(query)
        farmAvg["Benemilk Robo"] = farmAvg.apply( lambda row: self.getFeedForRow(self, row, "Benemilk Robo"), axis=1)
        return farmAvg
#        self.df = self.df.merge(df,fp, how='left', left_on='Date', right_on='Date')

#%% GAM
def tryGam(df):
    from pygam import LinearGAM
    cowGb = df.groupby(["Farm","Cow Code"])
    cowGb = cowGb.filter(lambda x: x.count() > 6)
    x = df.loc[:,"DIM"]
    y = df.loc[:,"Production today"]        
    gam = LinearGAM(n_splines=10).gridsearch(x,y)
    return gam

#%% - MAIN
#fp = FeedingPlan(df,51663)
#fa = fp.integrateFeedPlan()
#stats =df.describe(exclude=["object"])

#gam = tryGam(df)    




