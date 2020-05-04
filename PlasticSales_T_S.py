# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:50:55 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plastic=pd.read_csv("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\FORECASTING\PLASTIC SALES\PlasticSales.csv")
plastic.isnull().sum()
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
p=plastic["Month"][0]
p[0:3]
plastic["months"]=0

for i in range(60):
    p=plastic["Month"][i]
    plastic["months"][i]=p[0:3]
    
plastic["t"]=np.arange(1,61)
plastic["t_squared"]=plastic["t"]*plastic["t"]
plastic["log_sales"]=np.log(plastic["Sales"])

month_dummies=pd.get_dummies(plastic["months"])
month_dummies=month_dummies.iloc[:,[4,3,7,0,8,6,5,1,11,10,9,2]]

Plastic_sales=pd.concat([plastic,month_dummies],axis=1)
Plastic_sales["Sales"].plot()

#######Split ther data into train and test########
Train=Plastic_sales.head(40)
Test=Plastic_sales.tail(20)
Test=Test.set_index(np.arange(1,21))

############Building the Linear model############
import statsmodels.formula.api as smf
Lin_model=smf.ols("Sales~t",data=Train).fit()
Lin_pred=pd.Series(Lin_model.predict(pd.DataFrame(Test["t"])))
Lin_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Lin_pred))**2))#####248.92

###########Exponential###########
Exp_model=smf.ols("log_sales~t",data=Train).fit()
Exp_pred=pd.Series(Exp_model.predict(pd.DataFrame(Test["t"])))
Exp_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Exp_pred)))**2))#######250.10

##########Quadratic ############
Quad_model=smf.ols("Sales~t+t_squared",data=Train).fit()
Quad_pred=pd.Series(Quad_model.predict(Test[["t","t_squared"]]))
Quad_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Quad_pred))**2)) ######495.46

######Additive Seasonality#########
Add_model=smf.ols("Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
Add_pred=pd.Series(Add_model.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]))
Add_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Add_pred))**2))  ####263.23


######Additive Linear Seasonality#########
Add_Lin_model=smf.ols("Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t",data=Train).fit()
Add_pred_lin=pd.Series(Add_Lin_model.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","t"]]))
Add_Lin_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Add_pred_lin))**2))########105.24


###Additive Seasonality with quadratic trend###########
Add_Quad_model=smf.ols("Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t+t_squared",data=Train).fit()
Add_Quad_pred=pd.Series(Add_Quad_model.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","t","t_squared"]]))
Add_Quad_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Add_Quad_pred))**2))########118.23

#####Multiplicative Seasonality#################
Mul_model=smf.ols("log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
Mul_pred=pd.Series(Mul_model.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]))
Mul_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Mul_pred)))**2))#####266.61

###Multiplicative Additive Seasonality ###############
Mul_add_model=smf.ols("log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t",data=Train).fit()
Mul_add_pred=pd.Series(Mul_add_model.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","t"]]))
Mul_add_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Mul_add_pred)))**2))######117.11


###Multiplicative Additive Quadratic ###############
Mul_add_Quad_model=smf.ols("log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t+t_squared",data=Train).fit()
Mul_add_Quad_pred=pd.Series(Mul_add_Quad_model.predict(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","t","t_squared"]]))
Mul_add_Quad_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Mul_add_Quad_pred)))**2))######178.01

#########Storing the error values#########
data={"Model":pd.Series(["Lin_rmse","Exp_rmse","Quad_rmse","Add_rmse","Add_Lin_rmse","Add_Quad_rmse","Mul_rmse","Mul_add_rmse","Mul_add_Quad_rmse"]),"RMSE":pd.Series([Lin_rmse,Exp_rmse,Quad_rmse,Add_rmse,Add_Lin_rmse,Add_Quad_rmse,Mul_rmse,Mul_add_rmse,Mul_add_Quad_rmse])}
RMSE_Table=pd.DataFrame(data)

#####Additive with linear Seasonality gives low rmse value#######
#Final Model#######
final_model=smf.ols("Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t",data=Plastic_sales).fit()
final.head()
