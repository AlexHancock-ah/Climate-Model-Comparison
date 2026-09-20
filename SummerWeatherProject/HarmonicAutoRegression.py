#Main Project Code
import queue

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima import params
from statsmodels.tsa.statespace import kalman_filter
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

    for lag in range(1,lags):
        test_stat = pacf_vals[lag] * np.sqrt(n)
        if abs(test_stat) < 1.96:
            lag = lag + 1
        else:
            return lag
    return None


#Useful data frames to quote from.
weather = pd.read_csv('/Users/alexhancock/PycharmProjects/PythonProject/.venv/SouthamptonWeather.csv')
CleanWeather = weather.replace('---',None).dropna()
TempUnclean = weather.loc[:,['yyyy','mm','tmax','tmin']].replace('---',None)
dates = weather.loc[:,['yyyy','mm']]
tempo = pd.concat([dates, TempUnclean.apply(dropstar_tmin, axis=1), TempUnclean.apply(dropstar_tmax, axis=1)],axis=1,).astype(float)
temp = tempo.rename(columns={0:'tmin',1:'tmax'}).dropna()

###Quantifying climate change

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

## Data with incomplete years dropped
hr_data = drop_incomplete_years(temp,1855).reset_index(drop=True)


def Build_Matrix(t, K,P,trend,):
    n = len(t)
    cols = [np.ones(n)]          # intercept column
    if trend:
        cols.append(t.astype(float))  # linear trend term
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / P))
        cols.append(np.cos(2 * np.pi * k * t / P))
    return np.column_stack(cols)



class ModelResult:

    def __init__(self,fittedvalues,resid,params,quen_test_result,LogLikelihood,aic):
        self.fittedvalues = fittedvalues
        self.resid = resid
        self.params = params
        self.quen_test_result = quen_test_result
        self.aic = aic
        self.LogLikelihood = LogLikelihood

    def Summary(self):
        return f"Params = {self.params}\n AIC = {self.aic}\n"



def HAR(series,Period):

    ## it would be ideal for HAR to apply onto the same series as SARIMAX without the years removed
    ##



    ## initial definitions
    n = len(series)
    t = np.arange(1, n + 1)
    ## For main purposes K = 2 is reasonable without over fitting or underfitting components to data
    K = 2

    X = Build_Matrix(t,K,Period,trend=True)
    b, residual, rank, sv = np.linalg.lstsq(X, series, rcond=None)
    y_est = X @ b

    resid = pd.Series(series-y_est,index=series.index)
    ## Fitting AR to residuals.

    max_lags = np.floor(Period/2).astype(int)

    AR_p = Quen_test(resid,max_lags)
    ## lags chosen as period over 2 as otherwise could affect harmonic regression as K=6

    ##incorperating AR model into residuals that match with series index
    if AR_p > 0:
        AR_model = AutoReg(resid,lags=AR_p,old_names=False).fit()
        AR_fitted_full = AR_model.fittedvalues.reindex(series.index, fill_value=0)
    else:
        AR_fitted_full = np.zeros(len(series),dtype=float)



    ## MODEL RESULTS
    fittedvalues = pd.Series((AR_fitted_full + y_est),index=series.index)
    resid = series - fittedvalues
    params = b
    AR_p = AR_p

    ## Determining likelihood of model
    ## model currently = b[0] + b[1] + b[2]sin((2pi*K*t)/P) + b[3]cos((2pi*K*t)/P) + ... + AR_coeff*X(t-1) + AR_const
    num_params = len(b) + len(AR_model.params) + 1 ## 1 is for residual variance
    std = np.std(resid)
    LogLikelihood = -(n/2)*(np.log(2*np.pi) + 2*np.log(std)) - (0.5*np.square(std))*np.sum(np.square(resid))

    ## Determining AIC of the model
    AIC = 2*num_params - 2*LogLikelihood


    return ModelResult(resid = resid,
                        fittedvalues = fittedvalues,
                        params = params,
                        quen_test_result = AR_p,
                        LogLikelihood = LogLikelihood,
                        aic = AIC)





HAR_model = HAR(hr_data['tmax'],12)
print(HAR_model.resid.describe())



