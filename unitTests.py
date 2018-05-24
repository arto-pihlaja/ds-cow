# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:50:57 2018

@author: pihlaart1
"""

import unittest
import pandas as pd
import processUnifiedFile as cut
from datetime import datetime

class TestFarmAverages(unittest.TestCase):
    df = pd.DataFrame(None)
    def setUp(self):
        if self.df.shape[0] < 1:
            self.df = cut.loadAndCleanse(test=True)
        
    def testGrouping(self):
        gr = cut.buildFarmAverages(self.df)
        self.assertEqual(gr.shape[0],115, msg="Wrong number of rows in aggregate")
        
    def testNearest(self):
        now =  datetime.now()
        self.assertEqual( cut.FeedingPlan.nearest(self.df["Date"],now), datetime.strptime("20180507", "%Y%m%d"))

    def testFarmAvg(self):
        pass
        
        
if __name__ == '__main__':
    unittest.main()