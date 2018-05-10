# -*- coding: utf-8 -*-
"""
Created on Tue May  8 22:48:44 2018

@author: pihlaart1
"""
import os
import pandas as pd
os.chdir("C:\\Users\\pihlaart1\\Documents\\BilotGo\\R")
#headers = pd.read_csv(".\\data\\headers.csv",sep=";",header=None).T.iloc[1,:]
#tuotos = pd.read_csv("..\\Tuotantotutka_files\\TUOTOS_MILKING_ROBOT_20180508_134232.csv",sep=";",names=headers, nrows=200)
tuotos = pd.read_csv(".\\data\\tuotos_milking_robot_sample5000.csv",sep=";").iloc[:,1:]
tuotos.loc[:,"Date"] = pd.to_datetime(arg=tuotos.loc[:,"Date"],format="%Y%m%d") 
CowCodeListSeries = tuotos.loc[:,"CowCode"].str.split(",")
tuotos["CowNumber"] = CowCodeListSeries.apply(lambda list: list[0])
cowCodeDf = pd.DataFrame(item for item in CowCodeListSeries)
tuotos = pd.merge(tuotos,cowCodeDf,left_index=True,right_index=True)
del(CowCodeListSeries, cowCodeDf)
tuotos.sort_values(by="Date",inplace=True)

dateLimiter = (tuotos.loc[:,"Date"] > "2017-1-1")
tuotos = tuotos.loc[dateLimiter,:]
tuotos["Farm Price Per Liter"][tuotos["Farm Price Per Liter"]!=0].min()
print("Number of customers:", tuotos["Customer number"].nunique())
print("Number of observations per customer",tuotos["Customer number"].value_counts())
print("Number of observations per customer",tuotos.groupby("Customer number")["Production today"].value_counts())
dailyFarmStats = tuotos.groupby(["Customer number","Date"]).agg({
        "Days Milking":"mean",
        "Feeding total":"mean",
        "Production today":"mean"
        })
t = tuotos.loc[(tuotos['Customer number'] == 17356) & (tuotos['Date']=='2018-1-16')]
t["Days Milking"].sum() #There must be something wrong with for example this. All cows can't have 0!