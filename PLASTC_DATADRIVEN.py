# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:04:08 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as tsaplots
from datetime import datetime,time
import seaborn as sns
import statsmodels.api as sm

plastic=pd.read_csv("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\FORECASTING\PLASTIC SALES\PlasticSales.csv")
plastic.isnull().sum()

plastic["Date"]=pd.to_datetime(plastic["Month"].str.replace(r'-(\d+)$',r'-19\1'))
plastic["month"]=plastic.Date.dt.strftime("%b")
plastic["year"]=plastic.Date.dt.strftime("%Y")

heat_map=pd.pivot_table(data=plastic,values="Sales",index="month",columns="year",aggfunc="mean",fill_value=0)
sns.heatmap(heat_map,annot=True,fmt="g")

sns.boxplot(x="month",y="Sales",data=plastic)
sns.boxplot(x="year",y="Sales",data=plastic)

for i in range(2,24,6):
    plastic["Sales"].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=4)
seasonal_dec=sm.tsa.seasonal_decompose(plastic["Sales"],freq=3)
seasonal_dec.plot()

Train=plastic.head(48)
Test=plastic.tail(12)
Test=Test.set_index(np.arange(1,13))

#MAPE
def MAPE(pred,org):
    temp=np.abs((pred-org))*100/org
    return np.mean(temp)

###Simple exponential Smoothing#######
ses=SimpleExpSmoothing(Train["Sales"]).fit()
ses_pred=ses.predict(start=Test.index[0],end=Test.index[-1])
ses_MAPE=MAPE(ses_pred,Test.Sales)#######26.09

HW=Holt(Train["Sales"]).fit()
HW_pred=HW.predict(start=Test.index[0],end=Test.index[-1])
HW_Mape=MAPE(HW_pred,Test.Sales) #########26.60



HW_exp_add=ExponentialSmoothing(Train["Sales"],trend="add",seasonal="add",seasonal_periods=12,damped=True).fit()
HW_exp_pred=HW_exp_add.predict(start=Test.index[0],end=Test.index[-1])
HW_exp_add_Mape=MAPE(HW_exp_pred,Test.Sales)####25.50

HW_exp_mul_add=ExponentialSmoothing(Train["Sales"],trend="mul",seasonal="add",seasonal_periods=12).fit()
HW_exp_mul_pred=HW_exp_mul_add.predict(start=Test.index[0],end=Test.index[-1])
HW_exp_mul_Mape=MAPE(HW_exp_mul_pred,Test.Sales)####25.73

