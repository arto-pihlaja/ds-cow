# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 22:57:14 2018

@author: pihlaart1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def buildCowdata(fileName,existingDf):
    newData = pd.read_excel(fileName, decimal=",", sep=";", index_col=None, na_values=["NA"], encoding="latin1")
    newData = newData.iloc[3:,2:] #Remove empty rows and unnecessary columns
    newData = newData.dropna()
    if existingDf is None:
        existingDf = newData        
    else:
        newData = newData.iloc[1:,:] #remove header row from appended Df
        existingDf = existingDf.append(newData)
    return existingDf
        
try: 
    if cowdata.size < 1:
        pass
    else:
        pass        
        
except NameError: #cowdata does not exist yet, build it
        dataFiles = ["LR_51663_03012018.xls", 
                     "LR_51663_06022018.xlsx",
                     "LR_51663_06032017.xlsx",
                     "LR_51663_19012018.xls"]        
        cowdata = None
        for file in dataFiles:
            path = "./data/" + file
            cowdata = buildCowdata(path, cowdata)
        columnNamesFi = cowdata.iloc[0,:]
        cowdata = cowdata.iloc[1:,:] #Remove first row with Finnish column names
        columnNamesEn = ["CowName","ProdPeriod","DaysInMilk","MilkQuantity","MilkQntyDeviation","AvgMilkQnty","Milkings","PassAvg","KgConcentratedFoodPerKgMilk","FeedTotal","TodaysTotal","LeftoverTotal","Fat%","Protein%","FatToProteinRatio","MasticatingMinutes","AvgWeight","Cells"]
        cowdata.columns = columnNamesEn


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
