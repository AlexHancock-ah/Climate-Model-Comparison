from statsmodels.tsa import seasonal

import HarmonicAutoRegression as HAR
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import pmdarima
from pmdarima.arima import nsdiffs
from pmdarima.utils import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

##using the same temperature data of temperature max from MET office
##We must consider how to alter the data to deal with missing elements in the 1940s
temp_data = HAR.TempUnclean
SARIMA_data_temp = pd.concat([HAR.dates, temp_data.apply(HAR.dropstar_tmin, axis=1), temp_data.apply(HAR.dropstar_tmax, axis=1)],axis=1,).astype(float)
SARIMA_data = SARIMA_data_temp.rename(columns={0:'tmin',1:'tmax'})
# print(temp_data.loc[temp_data['yyyy'].isin([1941,1940,1944])])
## 1940, 1941, 1944 are missing a few months largest gap being 8 months and smaller 3 month gap in 1944
## 3 months could be filled in by my Harmonic and auto regression
## as it turns out SARIMA naturally handles these type of gaps because of the Kalman Filter
## which skips the contribution to the functions likelihood, keeps the seasonal element with AR in play
## and tackles the uncertainty the gap brings

## Useful Functions
def test_stationarity(series, regression,autolag):
    adf_result = adfuller(series,regression=regression,autolag=autolag)
    kpss_result = kpss(series,regression=regression)
    print(adf_result[0],adf_result[1],adf_result[4])
    print(kpss_result[0],kpss_result[1],kpss_result[3])

def lin_reg_matrix(series):
    t = np.arange(1,len(series)+1)
    n = len(t)
    cols = [np.ones(n), t.astype(float)]
    return np.column_stack(cols)

def remove_lin_trend(series):
    X = lin_reg_matrix(series)
    b,residual,rank,sv = np.linalg.lstsq(X,series,rcond=None)
    y_hat = X @ b
    return (series - y_hat),X

def Bartletts_test(series,lags,seasonal,step):
    n = len(series)
    acf_vals = acf(series,nlags=84,fft=True)
    if seasonal:
        seasonal_lags = list(range(step,lags,step))
        r_seasonal = [acf_vals[l] for l in seasonal_lags]

        for i, lag in enumerate(seasonal_lags):
            prior = r_seasonal[:i]
            se = np.sqrt((1/n) * (1 + 2*sum(r**2 for r in prior)))
            z = r_seasonal[i] / se
            print(f"lag {lag}:, acf value at lag {r_seasonal[i]}, sig={abs(z)>1.96}")
    elif not seasonal:
        lags = list(range(1,lags+1))
        r_lags = [acf_vals[l] for l in lags]
        for i, lag in enumerate(lags):
            prior = r_lags[:i]
            se = np.sqrt((1/n) * (1 + 2*sum(r**2 for r in prior)))
            z = r_lags[i] / se
            print(f"lag {lag}:, acf value at lag {r_lags[i]}, sig={abs(z)>1.96}")
    else:
        print('true false error on seasonal')

def Quenouilles_test(series,lags,seasonal):
    n = len(series)
    pacf_vals = pacf(series,nlags=lags,method= 'ywm')
    if seasonal:
        seasonal_lags = [12,24,36,48,60,72,84]

        SE = 1/np.sqrt(n)
        for lag in seasonal_lags:
            z = pacf_vals[lag]/SE
            print(f"lag {lag}: phi={pacf_vals[lag]:.4f}  z={z:.2f}  sig={abs(z)>1.96}")
    elif not seasonal:
        lags = list(range(1,lags+1))

        SE = 1/np.sqrt(n)
        for lag in lags:
            z = pacf_vals[lag]/SE
            print(f"lag {lag}: phi={pacf_vals[lag]:.4f}  z={z:.2f} sig={abs(z)>1.96}")
    else:
        print('true false error on seasonal')


## SARIMA takes inputs (p,d,q) - non-seasonal and (P,D,Q) Seasonal orders along with period 's' of the cycle
## firstly we will decide (d,D) as they are important so that we can assume stationarity along with the fact that s = 12
# test_stationarity(SARIMA_data['tmax'].dropna(),regression='ct',autolag='AIC')
## Using regression = 'c'
## Running this gives ADF -5.8 < -3.4 hence implying reject unit root concluding time series is stationary
## KPSS gives: 0.785 > 0.739 hence we reject H0 that series is stationary so, series is non-stationary
## Using regression = 'ct'
## Running this gives ADF -6.46 < -3.96 hence implying reject unit root concluding time series is stationary
## KPSS gives: 0.037 < 0.119(10%) hence we accept H0 as there is insufficient evidence to suggest h0 is not true
## Overall as we know from HAR that there is infact a linear trend so it is more relevent to assume that the data
## is trend-stationary

## Testing for D
## We use detrended data to avoid the tests getting interfered with
detrended_data = remove_lin_trend(SARIMA_data['tmax'].dropna())[0]
X = remove_lin_trend(SARIMA_data['tmax'].dropna())[1]
# OSCB_result = nsdiffs(detrended_data,test='ocsb',m=12,max_D=2)
# print(OSCB_result) # 0
# CH_result = nsdiffs(detrended_data,test='ch',m=12,max_D=2)
# print(CH_result) # 1

## doing this on the detrended data results in issues as it would difference some months that are not the same
## due to the 1940s gaps
# var = detrended_data.var()
# var_diff12 = detrended_data.diff(12).var()

# var = HAR.hr_data['tmax'].var()
# var_diff12 = HAR.hr_data['tmax'].diff(12).var()
# print(var)
# print(var_diff12)

##For the following tests I will be using data from HAR with the incomplete years dropped
##This is to allow for the following tests to not have lag values that are incorrect
##due to the shift that occours just by removing the months

data_Sbart = HAR.hr_data['tmax'].diff(12).dropna()

## Bartlett's test
# Bartletts_test(data_bart,84,seasonal=True,step=12)
## indicates first lag is significant giving Q = 1

tentative = SARIMAX(data_Sbart,order=(0,0,0),seasonal_order=(0,0,1,12)).fit()
data_S_quen = data_Sbart - tentative.fittedvalues

# Quenouilles_test(data_S_quen,84,seasonal=True)
## indicative of no correlation between lags so P = 0; will double check as it would make sense that there is
# Quenouilles_test(data_bart,84,True)
## Note how when Quenouilles test is run on the data for Bartlett's test every lag is significant
## this is a symptom of having an infinite AR representation of MA(1) hence why Bartlett's test was done first.

## Now the goal would be to find p and q the non-seasonal AR and MA components

data_bart = HAR.hr_data['tmax'].diff(12).dropna()
# Bartletts_test(data_bart,6,False,1) # testing for monthly correlation
## test indicates q = 3
tentative2 = SARIMAX(data_bart,order=(0,0,3)).fit()
data_quen = data_bart - tentative2.fittedvalues
# Quenouilles_test(data_quen,6,False)
# Bartletts_test(data_quen,6,True,1)

## The prior tests are indicative of using q = 3 and p = 0 or alternatively p = 2 and q = 0
Sarima_model = SARIMAX(SARIMA_data['tmax'],order=(0,0,3),seasonal_order=(0,1,1,12)).fit()
# plt.scatter(range(len(Sarima_model.fittedvalues)),Sarima_model.fittedvalues,color='blue')
# plt.scatter(range(len(SARIMA_data)),SARIMA_data['tmax'],color='red')
# plt.show()
# print(sum(np.abs(Sarima_model.resid.dropna())))
print(Sarima_model.summary())