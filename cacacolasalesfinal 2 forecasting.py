# import pacakegs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import monthly_returns_heatmap as mrh
from pmdarima import auto_arima 

# print dataset
co = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Forecasting\\CocacolaSalesRawData.csv")
# split dataset accordingly 
co['date']= pd.to_datetime(co['Quarter'].str.split('_').apply(lambda x: '-'.join(x[::-1])))

c=[]
for i in range(len(co)):
    k=str(co.iloc[i,2])
    c.append('19'+k[2:])
 
co.date=c   
co.date=pd.to_datetime(co.date)
co["year"] = co.date.dt.strftime("%Y")    # year extraction
co["month"] = co.date.dt.strftime("%b")   # month extraction
co.index=co.date

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(co["Sales"],model="additive",extrapolate_trend='freq')
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(co["Sales"],model="multiplicative")
decompose_ts_mul.plot()

# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=co,values="Sales",index=["year"],columns=['month'],aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Sales",data=co)
sns.boxplot(x="year",y="Sales",data=co)

# Line plot for Sales based on year  and for each month
sns.lineplot(x="year",y="Sales",hue="month",data=co)

# moving average for the time series to understand better about the trend character in dataset
co1=co["Sales"]
co1.plot()
co1.plot(label="org")
for i in range(2,24,6):
    co["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(co["Sales"],lags=10)
tsa_plots.plot_pacf(co["Sales"])

# splitting the data into Train and Test data 
Train = co.head(31)
Test = co.tail(11)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method   #seasonal = add
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)
# Simple Exponential Method   #seasonal = mul
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)

# Holt method 
hw_model_holt = Holt(Train["Sales"]).fit()
pred_hw_holt = hw_model_holt.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw_holt,Test.Sales)

# Lets us use auto_arima from p
auto_arima_model = auto_arima(Train["Sales"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
                
            
auto_arima_model.summary() 

# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )

# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=11))

# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Sales)
 
# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Sales"], label='Train',color="black")
plt.plot(Test.index, Test["Sales"], label='Test',color="blue")
plt.plot(pred_hwe_add_add.index, pred_hwe_add_add, label='SimpleExponential',color="green")
plt.plot(pred_hwe_mul_add.index, pred_hwe_mul_add, label='SimpleExponential',color="orange")
plt.plot(pred_hw_holt.index, pred_hw_holt, label='Holts_winter',color="red")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="Auto_Arima",color="grey")


























