# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 22:57:14 2018

@author: pihlaart1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime

def buildCowdata(fileName,existingDf):
    newData = pd.read_excel(fileName, decimal=",", sep=";", index_col=None, na_values=["NA"], encoding="latin1")
    newData = newData.iloc[3:,2:] #Remove empty rows and unnecessary columns
    newData = newData.dropna()
    newData["Date"] = pd.to_datetime(arg=fileName.split("_")[2].split(".")[0],format="%d%m%Y")        
    if existingDf is None:
        existingDf = newData        
    else:
        newData = newData.iloc[1:,:] #remove header row from appended Df
        existingDf = existingDf.append(newData)
#    existingDf["Date"] = datetime.strptime(fileName.split("_")[2],"%d%m%Y")
    
    return existingDf
        
# Only load files if the dataframe doesn't exist yet
try: 
    if cowdata.size < 1:
        pass
    else:
        pass        
        
except (AttributeError, NameError): #cowdata does not exist yet, build it
        dataFiles = ["LR_51663_03012018.xls", 
                     "LR_51663_06022018.xlsx",
                     "LR_51663_06032017.xlsx",
                     "LR_51663_19012018.xls",
                     "LR_51663_20022018.xlsx",
                     "LR_51663_20032018.xls"]        
        cowdata = None
        for file in dataFiles:
            path = "./data/" + file
            cowdata = buildCowdata(path, cowdata)
        columnNamesFi = cowdata.iloc[0,:]
        cowdata = cowdata.iloc[1:,:] #Remove first row with Finnish column names
        columnNamesEn = ["CowName","ProdPeriod","DaysInMilk","MilkQuantity","MilkQntyDeviation","AvgMilkQnty","Milkings","PassAvg",
                         "KgConcentratedFoodPerKgMilk","FeedTotal","TodaysTotal","LeftoverTotal","Fat%","Protein%","FatToProteinRatio","MasticatingMinutes","AvgWeight","Cells","Date"]
        cowdata.columns = columnNamesEn

def showBoxPlot(cowdata):
    fig = plt.figure()
    for k in range(1,len(cowdata.columns)):
    #    fig.suptitle(columnNamesEn[k])
        fig.add_subplot(9, 2, k)
        plt.ylabel(columnNamesEn[k])
    #    ax.boxplot(cowdata.iloc[:,k])
        cowdata.iloc[:,k] = pd.to_numeric(cowdata.iloc[:,k])
        cowdata.iloc[:,k].plot.box()
    plt.show()
    x = pd.to_numeric(cowdata.loc[:,"DaysInMilk"])
    y = pd.to_numeric(cowdata.loc[:,"MilkQuantity"])
    #z = np.polyfit(x,y,2)
    #p = np.poly1d(z)
    #plt.plot(x,p(x),"b-")
    #plt.scatter(x,y)
    #plt.show()

def categorizeByDaysInMilk(row):
    if row["DaysInMilk"]<30:
        return 30
    if row["DaysInMilk"]>=30 and row["DaysInMilk"]<60:
        return 60
    if row["DaysInMilk"]>=60 and row["DaysInMilk"]<90:
        return 90
    if row["DaysInMilk"]>=90 and row["DaysInMilk"]<150:
        return 150
    if row["DaysInMilk"]>=150 and row["DaysInMilk"]<250:
        return 250 
    if row["DaysInMilk"]>=250:
        return 300

#cowdata["Under60"] = (cowdata.loc[:,"DaysInMilk"] < 60).astype(int)    
cowdata["DIMCategory"] = cowdata.apply (lambda row: categorizeByDaysInMilk(row), axis = 1)
cowdata = pd.get_dummies(cowdata, prefix="DIM_under_",columns=["DIMCategory"],drop_first=True)
cowdata = pd.get_dummies(cowdata, prefix="ProdPer_",columns=["ProdPeriod"],drop_first=True)