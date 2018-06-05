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
import confidential as cf  #Value calculating formulas, not shared
import numpy as np

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

def discardAnomalies(df): 
#    dateLimiter = (df.loc[:,"Date"] > "2017-1-1")    
#    df = df.loc[dateLimiter]
    dimLimiter = (df.loc[:,"DIM"] < 500)    
    df = df.loc[dimLimiter]
    passLimiter = (df.loc[:,"Ohikulut 24 h"] < 10)
    df = df.loc[passLimiter]
    feedLimiter = (df.loc[:,"Syonti yhteensa"] < 20)
    df = df.loc[feedLimiter]    
#    pcLimiter = (df.loc[:,"PrevCalving"] < 11)
#    df = df.loc[pcLimiter]        
    caLimiter = (df.loc[:,"Calving No"] < 11)
    df = df.loc[caLimiter]            
    pdLimiter = (df.loc[:,"PrevDIM"] < 500)
    df = df.loc[pdLimiter]        
#    pmLimiter = (df.loc[:,"PrevLypsyja"] < 10)
#    df = df.loc[pmLimiter]        
#    pfLimiter = (df.loc[:,"PrevSyonti"] < 20)
#    df = df.loc[pfLimiter]        
    milkingLimiter = (df.loc[:,"Lypsyja Last 24 h"] < 6)
    df = df.loc[milkingLimiter]        
#    weightAvgLimiter = (df.loc[:,"Weight avg"] < 900)
#    df = df.loc[weightAvgLimiter]        
#    productionAvgLimiter = (df.loc[:,"Production avg"] < 80)
#    df = df.loc[productionAvgLimiter]            
    productionTodayLimiter = (df.loc[:,"Production today"] < 80)
    df = df.loc[productionTodayLimiter]                
    fatLimiter = (df.loc[:,"Content fat percent"] < 0.06)
    df = df.loc[fatLimiter]        
    proteinLimiter = (df.loc[:,"Content protein perc"] < 0.06)
    df = df.loc[proteinLimiter]            
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
    return farmAgg    

def buildFarmLevelData(df):
    df = buildFarmAverages(df)
    

#%% cell - help deciding which percentage to use
def countPercentages(df):
    print("Number of non-zero Content fat percent: ", (df["Content fat percent"]!=0).sum() )
    print("Number of non-zero Last Content Fat Percent: ", (df["Last Content Fat Percent"]!=0).sum() )
    print("Number of non-zero Content protein perc: ", (df["Content protein perc"]!=0).sum() )
    print("Number of non-zero Last Content Protein Percent: ", (df["Last Content Protein Percent"]!=0).sum() )    
    
#%% CRM data
def loadCrmData(test=False):
    if test:
        nrows = 10000
    else:
        nrows = None # Load all        
    df = pd.read_csv("..\\SAPAnalyticsReport.csv",sep=";",decimal=",", low_memory=False, nrows=nrows, encoding='latin1')            
#    df = df.loc[((df.loc[:,"Pääasiakkuus"]=="Nauta") | (df.loc[:,"Pääasiakkuus"]=="Not assigned")) ,:]
    query = "(Pääasiakkuus == 'Nauta' | Pääasiakkuus == 'Not assigned') & (Industry == 'Alkutuottaja' | Industry == 'Not assigned' )"
    df = df.query(query)
    df.replace(to_replace="Not assigned", value="",inplace=True)
    width = df.shape[1]    
    for col in range((width - 11),width):    
        # format numeric columns
        df.iloc[:,col] = df.iloc[:,col].apply( lambda x: str(x).replace(".",""))
        df.iloc[:,col] = df.iloc[:,col].apply( lambda x: str(x).replace(",","."))
        df.iloc[:,col] = pd.to_numeric(df.iloc[:,col], errors="coerce")

    for col in ["Emolehmätila", "Kioski", "Koneruokinta", "Kuivaruokinta", "Lihakarjatila", "Loppukasvattamotila",
                "Luomukarjatila","Lypsykarjatila", "Parsi", "Pihatto", "Premix-ruokinta", "Puolitiiviste", "Robotti",
                "Täysrehu", "Tiiviste", "Edelläkävijä"]:    
        # format Boolean values as binary
        df.loc[:,col] = df.loc[:,col].apply( lambda x: 1 if x == "YES" else 0)
        df.loc[:,col] = df.loc[:,col].astype("int64")

        
    df.loc[:,"External ID"] = df.loc[:,"External ID"].apply( lambda x: str(x).replace("#","")) # fix some wrong values
#    df.loc[:,"External ID"] = df.loc[:,"External ID"].apply( lambda x: int(x))
    df.loc[:,"External ID"] = pd.to_numeric(df.loc[:,"External ID"], errors="coerce")
    df.loc[:,"Account ID"] = pd.to_numeric(df.loc[:,"Account ID"], errors="coerce").astype("int64", errors="ignore")
    df = df.loc[:,["External ID","Account","Pääasiakkuus","Päätilatyyppi", "Industry", "Kioski", "Koneruokinta","Lihakarjatila", "Luomukarjatila", "Lypsykarjatila", 
                   "Parsi", "Pihatto", "Robotti", "Täysrehu", "Tiiviste", "Edelläkävijä", "Lehmien määrä", "Keskituotos (litraa)"]]    
    df = df.rename({"External ID":"Farm",
                    "Pääasiakkuus":"CustomerType",
                    "Päätilatyyppi":"FarmType",
                    "Kioski":"Kiosk",
                    "Koneruokinta":"AutomaticFeed",
                    "Lihakarjatila":"MeatProducer",
                    "Luomukarjatila":"OrganicFarm",
                    "Lypsykarjatila":"DairyFarm",
                    "Parsi":"Yoke",
                    "Pihatto":"FreeRangeFarm",
                    "Robotti":"Robot",
                    "Täysrehu":"FullFeed",
                    "Tiiviste":"ConcentratedFeed",
                    "Edelläkävijä":"Pioneer",
                    "Lehmien määrä":"NbOfCows",
                    "Keskituotos (litraa)":"AvgProductionLiters",
                    }, axis="columns")        

    return df


#%% prepare from Ashwin's cleansed data set 24.5.
def loadCleansedData(test=False):
    if test:
        nrows = 10000
    else:
        nrows = None # Load all        
    df = pd.read_csv("..\\V_RobotPlusOneCows-LROnly-Comma-MainValuesModel\\V_RobotPlusOneCows-LROnly-Comma-MainValuesModel.csv",sep=";",decimal=",", low_memory=False, nrows=nrows, encoding='latin1')            
    df.loc[:,"Date"] = pd.to_datetime(arg=df.loc[:,"Date"],format="%Y%m%d") 
    df = df.iloc[:,3:] #dropping the artificial keys
    df.sort_values(by="Date",inplace=True)    
# harmonise percent values to < 1, so 3.3 => 0.033. Remove outliers. Values > 0.08 (8%) must be errors.
    for col in ["ProteinPercent", "FatPercent", "NextProtein", "NextFat"]:
        df.loc[:,col] = df.loc[:,col].apply( lambda x: x if x < 1 else x/100)
        df.loc[:,col] = df.loc[:,col].apply( lambda x: x if x < 0.08 else None)
        # replace outliers with mean
        mu = df.loc[:,col].mean()
        rowsWithNone = df.loc[:,col].isnull()
        df.loc[rowsWithNone,col] = mu
        
    df = df.rename({"Lypsyjä Last 24 h":"MilkingsLast24h",
                    "Syönti yhteensä":"FeedTotal24h"
                    }, axis="columns")  
    printCleansedFacts(df)
    return df

def printCleansedFacts(df):
    print("Are these values OK, for example:")
    totalObs = df.shape[0]
    print("Total number of observations", totalObs)
    calv = sum(df["Calving No"]>8)
    print("Number of observations with more than 8 calvings: ", calv, " or ", round(calv/totalObs*100, 1), "% of the observations" )
    dim = sum(df.DIM>500)
    print("Number of observations with DIM > 500: ", dim, "or", round(dim/totalObs*100, 1), "% of the observations")
    print("Highest protein percentage: ", round(df.ProteinPercent.max()*100, 1), "%")
    print("Number of observations with protein > 5%: ", sum(df["ProteinPercent"]>0.05))

def mergeCrmData(df, crmDf):    
    df = df.merge(right=crmDf, how="left", on="Farm")
    i = df.loc[:,"OrganicFarm"].isnull()
    df = df[~i]
    width = df.shape[1]
    for col in range(width-13,width-1):
    # reformat from float to int (0 and 1)
        df.iloc[:,col] = df.iloc[:,col].astype("int64")
#    df.to_csv("RoboCrmMerge.csv",sep=";",decimal=",")
    return df

def prepareDataSet2wkValue():
    dummy()
    df = loadCleansedData()
    cdf = loadCrmData()
    print("Data loaded. Now merging CRM data...")
    df = mergeCrmData(df, cdf)
    del(cdf)
    return df

#%% feature engineering & train/test split
def encodeCyclicalValues(cyclicalSeries, numberOfPeriods):
        sin_feat = np.sin(cyclicalSeries/numberOfPeriods *2 *np.pi)
        cos_feat = np.cos(cyclicalSeries/numberOfPeriods *2 *np.pi)
        return sin_feat, cos_feat
    
def addFeatures(df):
    df["ValueToday"] = calculateValueGeneric(df["Production today"], df["FatPercent"], df["ProteinPercent"])
    df["NextValue"] = calculateValueGeneric(df.NextProdToday, df.NextFat, df.NextProtein)
    df["MonthSin"], df["MonthCos"] = encodeCyclicalValues(df.Month, 12)
    return df
    
def splitTrainTest(features, labels):
    from sklearn.model_selection import train_test_split
    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(features, labels, test_size=0.2)
    return trainFeatures, testFeatures, trainLabels, testLabels 

def prepareDataForPredictingNextValue():
    df = prepareDataSet2wkValue()
    df = addFeatures(df)
    return df

def dataframeToTrainTestData(df):
    y = np.array(df.loc[:, "NextValue"])
    X = df.drop("NextValue", axis="columns")
    trainX, testX, trainy, testy = splitTrainTest(X, y)    
    return trainX, testX, trainy, testy 

    
#%% GAM
def tryGam(df):
    from pygam import LinearGAM
    cowGb = df.groupby(["Farm","Cow Code"])
    cowGb = cowGb.filter(lambda x: x.count() > 6)
    x = df.loc[:,"DIM"]
    y = df.loc[:,"Production today"]        
    gam = LinearGAM(n_splines=10).gridsearch(x,y)
    return gam
    
#%% tree for 
def randomForest(numberOfTrees, df=pd.DataFrame()):
    if df.shape[0] < 1:
        df = prepareDataForPredictingNextValue()        
    features = ["Farm", "MonthSin", "MonthCos", "Calving No", "DIM", "MilkingsLast24h", "FeedTotal24h", 
                "Production avg", "FatPercent", "ProteinPercent", 
                "OrganicFarm", "Yoke", "FreeRangeFarm", "Robot", "NextValue"]
    dfForPrediction = df.loc[:,features]
    trainX, testX, trainy, testy = dataframeToTrainTestData(dfForPrediction)
    print("Data import and formatting complete. Now training Random Forest model.")
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=numberOfTrees)
    rf.fit(trainX, trainy)
    prediction = rf.predict(testX)
#    predictionME = round(np.mean(abs(prediction - testy)), 2)
#    print("Prediction mean error: ", predictionME)    
    baselinePrediction(df)
    # Get numerical feature importances (source: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)
    importances = list(rf.feature_importances_) 
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    return rf

def baselinePrediction(df):
    # baseline: guess the value just stays the same. Calculate it's error.
    baselineME = round(np.mean(abs(df.NextValue - df.ValueToday)), 2)
    print("Baseline mean error: ", baselineME)

def evaluateResults(prediction, testy):
    errors = abs(prediction - testy)
    predictionME = round(np.mean(errors), 5)
    print("Prediction mean error: ", predictionME)        
    
#%% cell - clustering cows by key attributes. Use elbow method to choose number of clusters. Then calculate Farm averages per cluster.

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
#    cluster by calculated value, calving no, DIM, lypsyjä, syönti yht. Problem: calving number is categorical rather than continuos. 
#       Kmeans expects continous
    features = ["Calving No", "DIM", "MilkingsLast24h", "FeedTotal24h", 
                "ValueToday"]
    df = df.loc[:,features]
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
    #calculate Farm averages by date and cluster
    aggregation = {"DIM":"mean",
                   "Production today":"mean",                   
                   "Production avg":"mean",
                   "Calving No":"mean",
                   "Lypsyja Last 24 h":"mean",
                   "Ohikulut 24 h":"mean",
                   "Syonti yhteensa":"mean",
                   "Cow Code":"count"}
    clusterAvgDf = df.groupby(["Farm","Date","CowCluster"], as_index=False).agg(aggregation)
# pick just a few dates for clarity    
    return clusterAvgDf

def doClustering(df=pd.DataFrame()):
    # Start with a dataframe with calculated values
    if df.shape[0] <1:
        df = loadCleansedData()   
        df = addFeatures(df)
    cdf = kElbow(df, 1, 8)
    istr = input("Enter number of clusters to use: ")
    idx = int(istr) - 1
    c = cdf.loc[idx,"cluster"]
    df["CowCluster"] = c.labels_
    plt.figure(3)
    plt.scatter(c.cluster_centers_[:,3],c.cluster_centers_[:,4])
    plt.title("KMeans cluster centroids, x = Feed total, y = Calculated value")
    plt.show() 
    return(df)       
    
#%% linear regressions
def linearModelForNextValue(df=pd.DataFrame()):
    from sklearn import linear_model
    if df.shape[0] < 1:
        df = prepareDataForPredictingNextValue() 
        df = addFeatures(df)    
    features = ["Farm", "MonthSin", "MonthCos", "Calving No", "DIM", "MilkingsLast24h", "FeedTotal24h", 
                "ValueToday", "OrganicFarm", "NextValue"]
    predDf = df.loc[:,features]
    predDf = normalizeFeatures(predDf)
    y = predDf.NextValue
    X = predDf.drop("NextValue", axis="columns")
    trainX, testX, trainy, testy = splitTrainTest(X, y)    
    lin = linear_model.LinearRegression()
    lin.fit(trainX, trainy)
    prediction = lin.predict(testX)
    evaluateResults(prediction, testy)
    baselinePrediction(df)
    print ("Let's try adding the square of DIM")
    X["DIM2"] = X["DIM"]*X["DIM"]    
    X = normalizeFeatures(X)
    trainX, testX, trainy, testy = splitTrainTest(X, y)
    lin2 = linear_model.LinearRegression()
    lin2.fit(trainX, trainy)
    prediction = lin2.predict(testX)
    evaluateResults(prediction, testy)   
    print("R2 of DIM2 model: ", lin2.score(testX, testy))
    print("Model coefficients: ")
    features[len(lin2.coef_)-1] ="DIM2"
    for i in range(0,len(lin2.coef_)):
        print(features[i], lin2.coef_[i])
    from sklearn.feature_selection import f_regression
    fr = f_regression(X, y)
    print()
    print("F-regression scores:")
    print("Feature, F-value,  p-value")
    for i in range(0,len(features)):
        print(features[i], fr[0][i], fr[1][i])

def linearModelForValue(df):
    from sklearn import linear_model
    features = ["Farm", "MonthSin", "MonthCos", "Calving No", "DIM", "MilkingsLast24h", "FeedTotal24h",
                "OrganicFarm", "ValueToday"]    
    predDf = df.loc[:,features]
    print(predDf.dtypes)
    predDf = normalizeFeatures(predDf)
    y = predDf.ValueToday
    X = predDf.drop("ValueToday", axis="columns")
    trainX, testX, trainy, testy = splitTrainTest(X, y)    
    lin = linear_model.LinearRegression()
    lin.fit(trainX, trainy)
    prediction = lin.predict(testX)
    evaluateResults(prediction, testy)
    print ("Let's try adding the square of DIM")
    X["DIM2"] = X["DIM"]*X["DIM"]    
    X = normalizeFeatures(X)
    trainX, testX, trainy, testy = splitTrainTest(X, y)
    lin2 = linear_model.LinearRegression()
    lin2.fit(trainX, trainy)
    prediction = lin2.predict(testX)
    evaluateResults(prediction, testy)   
    print("R2 of DIM2 model: ", lin2.score(testX, testy))
    print("Model coefficients: ")
    features[len(lin2.coef_)-1] ="DIM2"
    for i in range(0,len(lin2.coef_)):
        print(features[i], lin2.coef_[i])
    bl = df.ValueToday.mean()
    predDf["bl"]=bl
    predDf = normalizeFeatures(predDf.iloc[:,1:])
    evaluateResults(predDf.bl,testy)

def GLMwithGamma(df):
    # source: http://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html?highlight=glm
    # creds: https://stackoverflow.com/questions/41749167/glm-gamma-regression-in-python-statsmodels        
    import statsmodels.api as sm
    features = ["Farm", "MonthSin", "MonthCos", "Calving No", "DIM", "MilkingsLast24h", "FeedTotal24h",
                "OrganicFarm", "ValueToday"]    
    predDf = df.loc[:,features]
    predDf = normalizeFeatures(predDf)
    y = predDf.ValueToday
    X = predDf.drop("ValueToday", axis="columns")
    trainX, testX, trainy, testy = splitTrainTest(X, y)     
    model = sm.GLM(trainy, trainX, family=sm.families.Gamma(link=sm.genmod.families.links.identity)).fit()
    print(model.summary())
    prediction = model.predict(testX)
    evaluateResults(prediction, testy)
    return model

def prepareLargerDataSet():
    df = loadAndCleanse()
    df = df.rename({"Syonti yhteensa":"FeedToday",
                    "Yhteensa tanaan":"RecommendedFeed",
                    "Content fat percent":"FatPercent",
                    "Content protein percent":"ProteinPercent",
                    "Weight avg":"WeightAvg",
                    "Lypsyja Last 24 h":"Milkings24h"
                    }, axis="columns")            
    cdf = loadCrmData()
    print("Data loaded. Now merging CRM data...")
    df = mergeCrmData(df, cdf)
    features = ["DIM", "Calving No", "FeedToday", "FatPercent", "ProteinPercent", "WeightAvg", "OrganicFarm", "Production today"]
    df = df.loc[:,features]
    return df



def binContinuous(df):
    binProd = pd.cut(df.loc[:,"Production today"],10)
    return binProd

    

def tryNn(df):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    y = df.loc[:,"Production today"]
    X = df.drop(["FatPercent","ProteinPercent","Production today"], axis="columns")
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1],activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1)) #output
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X,y,verbose=1,epochs=20)
    return model
    
#%% - MAIN
#gam = tryGam(df)   




