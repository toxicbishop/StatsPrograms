import pandas as pd
import numpy as np
date_rng = pd.date_range(start='2023-01-01',end='2023-01-10',freq='D')
data=np.random.randint(0,100,size=(len(date_rng),2))
df=pd.DataFrame(data,columns=['A','B'],index=date_rng)
print("Original DataFrame")
print(df)
daily_mean=df.resample('D').mean()
print("Daily Mean")
print(daily_mean)
#Multivariate Data Frames
date_rng = pd.date_range(start='2023-01-01',end='2023-01-10',freq='D')
data={"Temperature": np.random.randint(20,30,size=(len(date_rng))),
      "Humidity":np.random.randint(30,70,size=(len(date_rng)))}
df_multivariate=pd.DataFrame(data,index=date_rng)
print("Multivariate Data Frame")
print(df_multivariate)
daily_mean_multivariate=df_multivariate.resample('D').mean()
print("Daily Mean for Multivariate")
print(daily_mean_multivariate)
#Forecasting Formats
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt 
model = ARIMA(df['A'],order=(1,1,1))
model_fit=model.fit()
forecast=model_fit.forecast(steps=5)
print("Forecast for next 5 days")
print(forecast)
plt.figure(figsize=(10,5))
plt.plot(df['A'],label="Historical Data")
plt.plot(pd.date_range(start=df.index[
1]+pd.Timedelta(days=1),periods=5),forecast,label="Forecast",color="red")
plt.title("Time Series Forecasting")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.savefig("outputs/03_forecast.png", dpi=150, bbox_inches="tight")
plt.close()