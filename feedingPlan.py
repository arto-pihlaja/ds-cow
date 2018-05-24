# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:48:45 2018

@author: pihlaart1
"""
import pandas as pd
import processUnifiedFile as pu

###
def loadFoodPlan():
    fp = pd.read_csv("..\\Tuotantotutka_files\\TUOTOS_FEEDING_PLAN_20180508_134717.csv",sep=";")
    columnNames = ["Date", "FeedType", "FeedName", "Farm", "X4", "X5", "X6",
                   "F10", "F15", "F20", "F25", "F30", "F35", "F40", "F45", "F50", "F55", "F60"] 
    fp.columns = columnNames            
    fp.loc[:,"Date"] = pd.to_datetime(arg=fp.loc[:,"Date"],format="%Y%m%d")
    return fp
    
def getFarmFp(fp, farm):
        fp = fp.loc[(fp["FeedType"]=="Aine, kiloja") & (fp["Farm"]== farm),:]    
        return fp

def countDayDiff(fpDate, dfRowDate):
        if dfRowDate < fpDate: 
            diff = 999 
        else: 
            diff = (dfRowDate - fpDate ).days
        return diff

def nearest(fpDates, dfRowDate):
# Foodplan must be made before the data row date, so fpDate < dfRowDate
    #    return min(items, key= lambda x: abs(x - pivot))
    return min(fpDates, key= lambda fpDate: countDayDiff(fpDate, dfRowDate))

def getFeedForRow(row, feed, fp):
    nearestFeedPlanDate = nearest(fp["Date"], row["Date"][0])
    feedPlanRow = fp.loc[( (fp["Date"]==nearestFeedPlanDate) & (fp["FeedName"]==feed) ),:]
    if row.iloc[5] < 10:
        feed = "F10"
    elif row.iloc[5] < 15:
        feed = "F15"
    elif row.iloc[5] < 20:
        feed = "F20"
    elif row.iloc[5] < 20:
        feed = "F20"
    elif row.iloc[5] < 25:
        feed = "F25"
    elif row.iloc[5] < 30:
        feed = "F30"
    elif row.iloc[5] < 35:
        feed = "F35"
    elif row.iloc[5] < 40:
        feed = "F40"
    elif row.iloc[5] < 45:
        feed = "F45"
    elif row.iloc[5] < 50:
        feed = "F50"
    elif row.iloc[5] < 55:
        feed = "F55"
    elif row.iloc[5] < 60:
        feed = "F60"        
                
#    elif row["Production today"] < 25:
#        feed = "F25"
    fr = feedPlanRow[feed]
    try:
        return fr.iloc[0]
    except (IndexError):
        pass

def integrateFeedPlan(df, fp, farm):   #    Feeding plan integration to farm level daily aggregates
    # limit the full cowdataset to farm dataset
    df = df.loc[(df.loc[:,"Farm"]==farm),:]    
    print("Total ", df.shape[0], " rows for farm ", farm)        
    farmAvg = pu.buildFarmAverages(df)
    print("Total ", farmAvg.shape[0], " summary rows for farm ", farm)     
    fp = getFarmFp(fp, farm)
#    farmAvg[feed] = farmAvg.apply( lambda row: getFeedForRow(row, feed, fp), axis=1)
    farmAvg["Benemilk Robo"] = farmAvg.apply( lambda row: getFeedForRow(row, "Benemilk Robo", fp), axis=1)
    farmAvg["Opti 40"] = farmAvg.apply( lambda row: getFeedForRow(row, "Opti 40", fp), axis=1)
    farmAvg["Barley"] = farmAvg.apply( lambda row: getFeedForRow(row, "Ohra, >62 kg/hl", fp), axis=1)
    farmAvg["Oats"] = farmAvg.apply( lambda row: getFeedForRow(row, "Kaura, >54 kg/hl", fp), axis=1)
    farmAvg["Wheat"] = farmAvg.apply( lambda row: getFeedForRow(row, "Vehnä, >72 kg/hl", fp), axis=1)
    return farmAvg
#        self.df = self.df.merge(df,fp, how='left', left_on='Date', right_on='Date')
