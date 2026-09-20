
# #Moving Average - before we fit the model
# #plotting residuals to determine weather a linear term is appropriate, (with without)
# X_1 = Build_Matrix(t,K,P,trend=True)
# b_hat,residual_T,rank_1,sv_1 = np.linalg.lstsq(X_1, hr_data['tmin'], rcond=None)
# print(residual_F,residual_T)
# # note how residual_T is significantly lower that without so a linear trend is good
# #is linear trend the best though

# ##Checking weather the winter heats up faster than the summer
# X_2 = Main.Build_Matrix(np.arange(0,852),2,12,trend=True)
# ##X_2 least square data for 1855 to 1920 ish
# b_2,residuals_2,rank_2,sv_2 = np.linalg.lstsq(X_2,hr_data['tmax'].iloc[:852])
# ## Least square data from 1920 to year 2000
# b_21,residuals_21,rank_21,sv_21= np.linalg.lstsq(X_2,hr_data['tmax'].iloc[852:1704])
# print(b_2 - b_21,b_2,b_21)
# print(residuals_2 - residuals_21)
#Note how b_2, b_21 have a negative difference on the second coefficient inferring that there is indeed a difference in slope between the two halves of data
#In addition the amplitude of the sinusoidal

# Testing for D using ACF plot
## ACF plot
# plot_acf(SARIMA_data['tmax'].dropna(),lags=60)
# plot_pacf(SARIMA_data['tmax'].dropna(),lags=60)
# D_ocsb = nsdiffs(SARIMA_data['tmax'].dropna(),m = 12, test='ocsb',max_D=2)
# D_ch = nsdiffs(SARIMA_data['tmax'].dropna(),m = 12, test='ch',max_D=2)
## at this point logic to cover later on will be differencing what ADF does how that ties into ocsb etc
# print(D_ocsb) # 0
# print(D_ch) # 1
## inconclusive
# var_D0 = SARIMA_data['tmax'].dropna().var()
# var_D1 = SARIMA_data['tmax'].dropna().diff(12).dropna().var()
# print(f"D0: {var_D0:.3f}, D1: {var_D1:.3f}, ratio: {var_D1/var_D0:.3f}")
## conclusive towards D = 1 as variance drops a large amount after differencing at 12
## d = 0, D = 1

## Identifying (P,Q) Seasonal components of AR and MA
y_diff = SARIMA_data['tmax'].diff(12).dropna()
#
# acf_vals, acf_confint = acf(y_diff,nlags=72,alpha=0.05,fft=True)
# pacf_vals, pacf_confint = pacf(y_diff,nlags=72,alpha=0.05,method='ywm')
#
seasonal_lags = [12,24,36,48,60,72]
# for lag in seasonal_lags:
#     sig_acf = not (acf_confint[lag, 0] <= 0 <= acf_confint[lag, 1])
#     sig_pacf = not (pacf_confint[lag, 0] <= 0 <= pacf_confint[lag, 1])
#     print(lag, "ACF sig:", sig_acf, "PACF sig:", sig_pacf) # outputs weather ACF is significant at each lag
##indicating weather to use for acf Q = 1,2 etc. and for pacf what coefficient to use for P
##note while PACF is statistically significant weather it is practically significant to include lags from all elements
##is not necessarily important as due to the nature of the sample such results could arise from white noise
# plt.plot(abs(pacf_vals))
# plt.show()
## Importantly as MA(1) can be represented as an infinite AR series (geom)
# seasonal_pacf = [pacf_vals[l] for l in seasonal_lags]
# ratios = [seasonal_pacf[i+1]/seasonal_pacf[i] for i in range(len(seasonal_pacf)-1)]
# print(ratios)
# for lag in seasonal_lags:
#     print(pacf_vals[lag])
## the prior indicates this is not the case hence the differencing term is not quite correct.
## repeating for D = 2
# y_diff2 = SARIMA_data['tmax'].diff(12).diff(12).dropna()
# acf_vals, acf_confint = acf(y_diff2,nlags=72,alpha=0.05,fft=True)
# pacf_vals, pacf_confint = pacf(y_diff2,nlags=72,alpha = 0.05, method='ywm')
# for lag in seasonal_lags:
#     print(pacf_vals[lag])
# seasonal_pacf = [pacf_vals[l] for l in seasonal_lags]
# ratios = [seasonal_pacf[i+1]/seasonal_pacf[i] for i in range(len(seasonal_pacf)-1)]
# print(ratios)
# print(y_diff.var(),y_diff2.var())
## variance jump is indicative of over differencing, hence due to all previous tests having contradictions
## I'm going to check weather I need a non-periodic differencing term in more detail as that could affect the structure
## leading to non stationarity
## for future diagnostic tests use coding with functions to make the process cleaner

#using this function to check stationarity of the single time differenced series.

##HURN DATA
data = pd.read_csv('/Users/alexhancock/PycharmProjects/PythonProject/.venv/HurnData.csv',skiprows=1)
# print(data.columns.tolist())

## finding yearly averages of max temp Hern data
Hurn_data = drop_incomplete_years(data.loc[:,['yyyy','mm','tmax']],date=1957)

# print(len(Hurn_data))
# print(Hurn_data['yyyy'].loc[Hurn_data['yyyy']==2026]) ## 2026 year dropped is inconsequential to purpose

yearly_mean_hurn = Hurn_data.groupby('yyyy')['tmax'].mean()
yearly_mean_southampton = HAR.hr_data.groupby('yyyy')['tmax'].mean()
comparison_data1 = yearly_mean_southampton.loc[yearly_mean_southampton.index >= 1957]
comparison_data2 = yearly_mean_hurn.loc[yearly_mean_hurn.index <= 1999]

# print(sum(abs(comparison_data1 - comparison_data2)))
## the yearly difference over the 42 years overlap that the Hurn and Southampton stations have their
## yearly means differ by 17.7 degrees C total



#### ALL the process behind HAR moved for organizing and cleaning up code.



#Main Project Code
import queue

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima import params
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf

#Useful functions for cleaning data
def dropstar_tmin(row):
    if isfloat(row['tmin']) == False:
        return row['tmin'].strip('*')
    else:
        return row['tmin']
def dropstar_tmax(row):
    if isfloat(row['tmax']) == False:
        return row['tmax'].strip('*')
    else:
        return row['tmax']

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def runsum (df,column,power):
    return df[column]*df.index**power

def Quen_test(series,lags):
    n = len(series)
    pacf_vals = pacf(series,nlags=lags,method='ywm')

    for lag in range(1,lags+1):
        test_stat = pacf_vals[lag] * np.sqrt(n)
        if abs(test_stat) < 1.96:
            return lag
        else:
            lag = lag + 1




#Useful data frames to quote from.
weather = pd.read_csv('/Users/alexhancock/PycharmProjects/PythonProject/.venv/SouthamptonWeather.csv')
CleanWeather = weather.replace('---',None).dropna()
TempUnclean = weather.loc[:,['yyyy','mm','tmax','tmin']].replace('---',None)
dates = weather.loc[:,['yyyy','mm']]
tempo = pd.concat([dates, TempUnclean.apply(dropstar_tmin, axis=1), TempUnclean.apply(dropstar_tmax, axis=1)],axis=1,).astype(float)
temp = tempo.rename(columns={0:'tmin',1:'tmax'}).dropna()

###Quantifying climate change

## For Time Series Analysis one can perhaps try to model the plot bellow using a trend component using Least Squares Regression

#Clearly in such a plot there is a substantial amount of noise and seasonality as such the use of LSR is not accurate
#however it should suffice as practice of coding and maybe show some general behavior
#Note Data from some months in 1940,41,44 are missing , may be necessary to omit entire years later on

# LSR_data = temp.loc[:,['tmin','tmax']]
# TmaxSum0 = LSR_data['tmax'].sum()
# TmaxSum1 = runsum(LSR_data,'tmax',1).sum()
# IndSum = sum(range(0,len(LSR_data)))
# ISS = IndSum*IndSum
# b = np.array([[TmaxSum0],[TmaxSum1]])
# A = np.array([[len(LSR_data),IndSum],[IndSum,ISS]])
# alpha = np.linalg.inv(A) @ b
# x = LSR_data.index
# plt.plot(x, (alpha[0] + (x*alpha[1])),color='red')
# plt.scatter(x,LSR_data['tmax'])
# plt.show()

#Curved line of best fit --- we have some unknown errors in here
# LSR_data = temp.loc[:,['tmin','tmax']]
# TmaxSum0 = LSR_data['tmax'].sum()
# TmaxSum1 = runsum(LSR_data,'tmax',1).sum()
# TmaxSum2 = runsum(LSR_data,'tmax',2).sum()
# IndSum = sum(range(0,len(LSR_data)))
# ISS = IndSum*IndSum
# b = np.array([[TmaxSum0],[TmaxSum1],[TmaxSum2]])
# A = np.array([[len(LSR_data),IndSum,ISS],[IndSum,ISS,ISS*IndSum],[ISS,ISS*IndSum,ISS*ISS]]).astype(float)
# alpha = np.linalg.inv(A) @ b
# x = LSR_data.index
# print(alpha)
# plt.scatter(x,LSR_data['tmax'])
# plt.show()

#Harmonic Regression
#before we start harmonic regression we must look at the missing data and either fill in values or remove the years with missing values
#otherwise the period will not be consistent with the data.
def drop_incomplete_years (df,date):
    for date in df['yyyy'].tolist():
        if len(df[df['yyyy']==date]) != 12:
            df = df[df['yyyy'] != date]
            date = date + 1
        else:
            date = date + 1
    return df


hr_data = drop_incomplete_years(temp,1855).reset_index(drop=True)
## Building matrix
n = 1704                  # number of observations
t = np.arange(1, n + 1)   # time index t = 1, ..., 1704
P = 12
K = 2 #nyquist limit is 6 which is un necessary for temprature


def Build_Matrix(t, K,P,trend,):
    n = len(t)
    cols = [np.ones(n)]          # intercept column
    if trend:
        cols.append(t.astype(float))  # linear trend term
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / P))
        cols.append(np.cos(2 * np.pi * k * t / P))
    return np.column_stack(cols)
X_0 = Build_Matrix(t,K,P,trend=True)
b,residual_F,rank,sv = np.linalg.lstsq(X_0, hr_data['tmax'], rcond=None)
y_hat = X_0 @ b

#this just fits a harmonic regression to the data

# model = sm.OLS(hr_data['tmax'],X_0).fit()
# print(model.summary())
#Summary data x1 indicates that as the coef >0 and the 95% confidence interval is also above 0
#[0.000,0.001] it is reasonable to conclude that there is in fact an upward slope of 0.0005 per month from 1855

## Modeling Residuals
## this model works currently as y_hat = b0 + b1t + (harmonic terms) calculated by least squares
## doesn't explain the residuals (what is left over)
# X_1 = Build_Matrix(t,K,P,trend=True)
# b_hat,residual_T,rank_1,sv_1 = np.linalg.lstsq(X_1, hr_data['tmin'], rcond=None)
residuals = pd.Series(hr_data['tmax']-y_hat,index=hr_data.index)
# plt.scatter(range(1,len(residuals)+1),residuals)
# plt.show()

## Plotting Auto Correlation Function (ACF) to determine weather there is structure to residuals data
## ACF plot
# fig, ax = plt.subplots(figsize=(10,10))
# plot_acf(residuals,lags=40,ax=ax)
# ax.set(title='ACF plot')
# ax.set_xlabel('lags')
# ax.set_ylabel('Auto Correlation')
# ax.axvline(x=12, color='red', linestyle='dashed', alpha=0.5)
# ax.axvline(x=24, color='red', linestyle='dashed', alpha=0.5)
# plt.tight_layout()
# plt.show()


## ACF plot indicates that there is marginal correlation at lag 1 and 2
## to fix we add an Auto Regressive model AR(2) probably to model residuals
AR_mod = AutoReg(residuals,lags=2, old_names=False).fit()
## Lag creates a gap of 2 due to nature of Auto regression hence im going to fill these missing values with the residuals of 1855 months 1 and 2
AR_fitted_full = AR_mod.fittedvalues.reindex(hr_data.index, fill_value=0)
HAR_model = pd.Series(y_hat + AR_fitted_full,index=hr_data.index)
##this demonstrates how my model fails to model anomalies
# plt.scatter(range(len(hr_data['tmax'])),hr_data['tmax'],c='b')
# plt.scatter(range(len(HAR_model)),HAR_model,c='r')
# plt.show()

resid_total = sum(abs(HAR_model - hr_data['tmax']))
# print(resid_total)