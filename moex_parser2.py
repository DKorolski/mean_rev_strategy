import asyncio
import aiohttp
import aiomoex
import os
import pandas as pd
import requests


filepath =os.getcwd()

def moex_candles(security,interval,start,end):
    async def main():
        async with aiohttp.ClientSession() as session:
            data = await aiomoex.get_board_candles(session=session, engine='futures', market='forts', board='TQBR', security=security, interval=interval, start=start, end=end)
            df = pd.DataFrame(data)
            df.set_index('begin', inplace=True)
            df.columns=['Open','Close','High','Low','Value','Volume','end']
            df=df[['Open','Close','High','Low','Volume']]
            df['Adj Close']=df['Close']
            df.index.names = ['Date']
            os.chdir(filepath)
            df.to_csv(security+'_'+start+'.csv')
    asyncio.run(main())
    filename = security+'_'+start+'.csv'
    df=pd.read_csv(filename, parse_dates=True, index_col=[0])
    os.remove(filename)
    return df

def moex_candles_stock(security,interval,start,end):
    async def main():
        async with aiohttp.ClientSession() as session:
            data = await aiomoex.get_board_candles(session=session,  security=security, board='TQBR', interval=interval, start=start, end=end)
            df = pd.DataFrame(data)
            df.set_index('begin', inplace=True)
            df.columns=['Open','Close','High','Low','Value','Volume','end']
            df=df[['Open','Close','High','Low','Volume']]
            df['Adj Close']=df['Close']
            df.index.names = ['Date']
            os.chdir(filepath)
            df.to_csv(security+'_'+start+'.csv')
    asyncio.run(main())
    filename = security+'_'+start+'.csv'
    df=pd.read_csv(filename, parse_dates=True, index_col=[0])
    os.remove(filename)
    return df

def candles_resample(df, interval):
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date') 
    df_resampled=df.resample(interval).agg({
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min',
        'Volume': 'sum'
        })
    # Drop rows with all NaN values if any
    df_resampled.dropna(how='all', inplace=True)
    return df_resampled


def moex_candles_index(security,interval,start,end):
    async def main():
        async with aiohttp.ClientSession() as session:
            data = await aiomoex.get_board_candles(session=session, engine='stock', market='index', security=security, interval=interval, start=start, end=end)
            df = pd.DataFrame(data)
            df.set_index('begin', inplace=True)
            df.columns=['Open','Close','High','Low','Value','Volume','end']
            df=df[['Open','Close','High','Low','Volume']]
            df['Adj Close']=df['Close']
            df.index.names = ['Date']
            os.chdir(filepath)
            df.to_csv(security+'_'+start+'.csv')
    asyncio.run(main())
    filename = security+'_'+start+'.csv'
    df=pd.read_csv(filename, parse_dates=True, index_col=[0])
    os.remove(filename)
    return df


def moex_candles_option(security,interval,start,end):
    async def main():
        async with aiohttp.ClientSession() as session:
            data = await aiomoex.get_board_candles(session=session, engine='futures', market='options', security=security, interval=interval, start=start, end=end)
            df = pd.DataFrame(data)
            df.set_index('begin', inplace=True)
            df.columns=['Open','Close','High','Low','Value','Volume','end']
            df=df[['Open','Close','High','Low','Volume']]
            df['Adj Close']=df['Close']
            df.index.names = ['Date']
            os.chdir(filepath)
            df.to_csv(security+'_'+start+'.csv')
    asyncio.run(main())
    filename = security+'_'+start+'.csv'
    df=pd.read_csv(filename, parse_dates=True, index_col=[0])
    os.remove(filename)
    return df