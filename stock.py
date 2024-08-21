import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

class Stock:
    def __init__(self):
        data = None
        
    def get_stock_data(self, ticker: str, period: str) -> pd.DataFrame:
        '''
        ticker: str - the stock ticker to get data for
        returns: pd.DataFrame - the stock data for the given ticker
        '''
            
        data = yf.Ticker(ticker).history(period=period) 
        
        # exponential smoothing 
        for col in data.columns:
            if col != 'Target':
                data[col] = SimpleExpSmoothing(data[col]).fit(smoothing_level=0.7, optimized=False).fittedvalues
        
        data = self.rsi(data)
        data = self.macd(data)
        data = self.price_rate_of_change(data)
        data = self.stochastic_oscillator(data)
        data = self.william_percent_r(data)
        data = self.on_balance_volume(data)
        data = self.moving_average(data)
        data = self.target(data)
        
        data.to_json(f'data/{ticker}.json')
        predicators = ['RSI_Signal', 'MACD_Signal', 'Price_Rate_Of_Change', 'Stochastic_Signal_Fast', 'Stochastic_Signal_Slow', 'William_Percent_R_Signal', 'On_Balance_Volume_Signal', 'SMA_Signal','Target']
        data.dropna(inplace=True)
        self.data = data[predicators]
        
        return self.data
    
    def moving_average(self, data: pd.DataFrame) -> pd.DataFrame:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        data['SMA_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, np.where(data['SMA_50'] < data['SMA_200'], -1, 0))
            
        return data
    
    def rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Delta'] = data['Close'].diff(1)
        data['Gain'] = data['Delta'].clip(lower=0).round(2)
        data['Loss'] = data['Delta'].clip(upper=0).abs().round(2)
        
        data['Avg_Gain'] = data['Gain'].rolling(window=14, min_periods=14).mean()[:14+1]
        data['Avg_Loss'] = data['Loss'].rolling(window=14, min_periods=14).mean()[:14+1]
        
        avg_gain = data['Avg_Gain'].copy()
        avg_loss = data['Avg_Loss'].copy()
        
        for i in range(14, len(data['Avg_Gain'])-1):
            avg_gain.iloc[i+1] = (avg_gain.iloc[i] * 13 + data['Gain'].iloc[i+1]) / 14
            avg_loss.iloc[i+1] = (avg_loss.iloc[i] * 13 + data['Loss'].iloc[i+1]) / 14  
        
        data['Avg_Gain'] = avg_gain
        data['Avg_Loss'] = avg_loss  
        
        data['RS'] = data['Avg_Gain'] / data['Avg_Loss']
        
        data['RSI'] = 100 - (100 / (1.0 + data['RS']))
        
        data['RSI_Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
            
        return data

    def macd(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Ema12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['Ema26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        data['MACD'] = data['Ema12'] - data['Ema26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        data['MACD_Signal'] = np.where(data['MACD'] > data['Signal'], 1, np.where(data['MACD'] < data['Signal'], -1, 0))
        
        return data

    def price_rate_of_change(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Price_Rate_Of_Change'] = (data['Close'] - data['Close'].shift(7)) / data['Close'].shift(7) * 100
        
        return data

    def stochastic_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Lowest_Low'] = data['Low'].rolling(window=14).min()
        data['Highest_High'] = data['High'].rolling(window=14).max()
        
        data['Fast_K'] = 100 * ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']))
        
        data['Fast_D'] = data['Fast_K'].rolling(window=3).mean().round(2)
        
        data['Slow_K'] = data['Fast_D']
        data['Slow_D'] = data['Slow_K'].rolling(window=3).mean().round(2)
        
        data['Stochastic_Signal_Fast'] = np.where(data['Fast_K'] > 80, -1, np.where(data['Fast_K'] < 20, 1, 0))
        data['Stochastic_Signal_Slow'] = np.where(data['Slow_K'] > 80, -1, np.where(data['Slow_K'] < 20, 1, 0))
        
        return data

    def william_percent_r(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Lowest_Low'] = data['Low'].rolling(window=14).min()
        data['Highest_High'] = data['High'].rolling(window=14).max()
        
        data['William_Percent_R'] = ((data['Highest_High'] - data['Close']) / (data['Highest_High'] - data['Lowest_Low'])) * -100
        
        data['William_Percent_R_Signal'] = np.where(data['William_Percent_R'] > -20, -1, np.where(data['William_Percent_R'] < -80, 1, 0))
        
        return data

    def on_balance_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        data['On_Balance_Volume'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'] + data['Volume'].shift(1), np.where(data['Close'] < data['Close'].shift(1), data['Volume'] - data['Volume'].shift(1), 0))
        
        data['On_Balance_Volume_Signal'] = np.where(data['On_Balance_Volume'] > data['On_Balance_Volume'].shift(1), 1, np.where(data['On_Balance_Volume'] < data['On_Balance_Volume'].shift(1), -1, 0))
        
        return data
    
    def target(self, data: pd.DataFrame) -> pd.DataFrame:
        target_val = data['Delta']
        target_val = np.where(target_val > 0, 1, 0)
        data['Target'] = target_val
                
        return data