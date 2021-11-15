import pandas as pd
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import numpy as np

from alpha_vantage.timeseries import TimeSeries

#api_key = 'RNZPXZ6Q9FEFMEHM'

#period = 60

#ti = TechIndicators(key=api_key, output_format ='pandas')

#data_ti, meta_data_ti = ti.get_rsi(symbol='MSFT', interval= '1min',
                                  #  time_period = period, series_type= 'close') # Relative Strength Index (Avg.upward price change/Avg.downward price change)

#data_sma, meta_data_sma = ti.get_sma(symbol='MSFT', interval= '1min',
                           #         time_period = period, series_type= 'close') # Simple moving average

#df1 = data_sma.iloc[1::] # на 1 значение больше, поэтому пропуск первого.
#df2 = data_ti
#df1.index = df2.index

#fig, ax1 = plt.subplots()
#ax1.plot(df1, 'b-')
#ax2 = ax1.twinx()
#ax2.plot(df2, 'r.')
#plt.title('SMA & RSI graph')
#plt.show()


# Add Time series example
ts = TimeSeries(key='RNZPXZ6Q9FEFMEHM',output_format='pandas')
#data, meta_data = ts.get_intraday(symbol='MCD',interval='60min', outputsize='full')
data, meta_data = ts.get_daily_adjusted(symbol='MCD', outputsize='compact')
#print(data)
data.to_excel('чёкого.xlsx')
df = pd.read_excel('чёкого.xlsx')
#data['4. close'].plot()
#plt.title('Intraday TimeSeries Microsoft')
#plt.show()
#print(df)
# нужно понять, какие интервавы в data и показателях выставлять, т.к. при изменении график ниебически прыгает
# Calculate the Exponential moving average (12 дней * 16 часов) 
ShortEMA = df['4. close'].ewm(span=12, adjust = False).mean()
# Calculate the Long term exponentialmoving average (26 дней)
LongEMA = df['4. close'].ewm(span=26, adjust = False).mean()
# индикатор схождения-расхождения скользящих средних (Moving Average Convergence/Divergence)
MACD = ShortEMA - LongEMA
# Calculate the signal line
signal = MACD.ewm(span = 9, adjust = False).mean()

# plot the chart
#plt.figure(figsize = (12.2, 4.5))
#plt.plot(df['4. close'].index, MACD, label = 'MSFT MACD', color = 'red')
#plt.plot(df['4. close'].index, signal, label = 'Signal line', color ='blue')
#plt.xticks(rotation = 45)
#plt.legend(loc = 'upper left')
#plt.show()

#создание новых колонок
df['MACD'] = MACD
df['Signal Line'] = signal
max_row = len(df.axes[0]) # number of rows


# функция, сигнализирующая к продаже
def buy_sell(DATASET):
    Buy = []
    Sell = []
    flag = -1
    
    for i in range (0, max_row): # сохраняется Цена при первом пересечении MACD и Сигнала, иначе NaN.
      if DATASET['MACD'][i] < DATASET['Signal Line'][i]:
        Sell.append(np.nan)
        if flag != 1:
            Buy.append(DATASET['4. close'][i])
            flag = 1
        else:
            Buy.append(np.nan)
      elif DATASET['MACD'][i] > DATASET['Signal Line'][i]:
        Buy.append(np.nan)
        if flag != 0:
            Sell.append(DATASET['4. close'][i])
            flag = 0
        else:
            Sell.append(np.nan)
      else:
        Buy.append(np.nan)
        Sell.append(np.nan)

    return (Buy, Sell)


# создание колонок buy и sell (здесь этот пидор из функции buy_sell аргументы распихивает на 2 колонки)
a = buy_sell(df)
df['Buy_Signal_Price'] = a[0]
df['Sell_Signal_Price'] = a[1]
print(df)
# Показываем значки на графике


df = df.set_index('date')


plt.subplot(2, 1, 2)
plt.plot(df['MACD'], label = 'MACD')
plt.plot(df['Signal Line'], label = 'Signal Line')
plt.legend(loc = 'upper left')

plt.subplot(2, 1, 1)
plt.plot(df['4. close'], label = 'Close Price', alpha = 0.35)

#plt.figure(figsize = (12.5, 4.5))
plt.scatter(df.index, df['Buy_Signal_Price'], color = 'green', label = 'Buy', marker = '^', alpha =1)
plt.scatter(df.index, df['Sell_Signal_Price'], color = 'red', label = 'Sell', marker = 'v', alpha =1)
#plt.plot(df['4. close'], label = 'Close Price', alpha = 0.35)
plt.title('MCD-RM: Close Price Buy & Sell Signals')
plt.xticks(rotation = 45)
plt.xlabel('Date')
plt.ylabel('Close Price SEK')
plt.legend(loc = 'upper left')

plt.tight_layout()
plt.show()