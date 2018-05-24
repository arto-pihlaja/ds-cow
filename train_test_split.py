# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:30:34 2018

@author: pihlaart1
"""
### add binary columns barley, oats, wheat, Benemilk Robo, 
import numpy as np
import pandas as pd

tdf = pd.read_csv("..\\UnifiedPeakPredictorWithSlope.csv",sep=";",encoding="latin1",low_memory=False)
mask = np.random.rand(tdf.shape[0]) < 0.9
train = tdf[mask]
test = tdf[~mask]
train.to_csv("..\\UPPTrain.csv",sep=";", decimal=".")
test.to_csv("..\\UPPTest.csv",sep=";", decimal=".")

