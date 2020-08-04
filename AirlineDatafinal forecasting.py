# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import monthly_returns_heatmap as mrh
# print dataset
Airline = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Forecasting\\AirlinesData.csv")
# split dataseet into time, data, year
Airline.index = pd.to_datetime(Airline.Month,format="%b-%y")
Airline.columns
Air1=Airline["Passengers"]
Air1.plot()

Airline["Date"] = pd.to_datetime(Airline.Month,format="%b-%y")
Airline["month"] = Airline.Date.dt.strftime("%b")
Airline["year"] = Airline.Date.dt.strftime("%Y")
Airline.month = (pd.to_datetime(Airline.month, format='%b'))
# plor heatmap
heatmap_y_month = pd.pivot_table(data=Airline,values="Passengers",index=["year"],columns=['month'],aggfunc="mean",fill_value=0)
heatmap_y_month.columns = heatmap_y_month.columns.strftime('%b')
sns.heatmap(heatmap_y_month,annot=True,fmt="g")
# plot boxplot
sns.boxplot(x="month",y="Passengers",data=Airline)
sns.boxplot(x="year",y="Passengers",data=Airline)
#plot lineplot
sns.lineplot(x="year",y="Passengers",hue="month",data=Airline)
# moving average for the time series to understand better about the trend character in dataset
Air1.plot(label="org")
for i in range(2,24,6):
    Airline["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(Airline["Passengers"],model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Airline["Passengers"],model="multiplicative")
decompose_ts_mul.plot()
# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(Airline["Passengers"],lags=10)
tsa_plots.plot_pacf(Airline["Passengers"])
#split dataset into train & test
Train = Airline.head(80)
Test = Airline.tail(16)
# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)
# Holt method
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)
# Simple Exponential Method   #seasonal = add
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers)
# Simple Exponential Method   #seasonal = mul
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)

# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")














