
import gc, argparse, sys, os, errno
import numpy as np
import pandas as pd
import seaborn as sns
#sns.set()
#sns.set_style('whitegrid')
import h5py
from PIL import Image
import os
from tqdm import tqdm as tqdm
import scipy
import sklearn
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--text',  type=str,
                    help='text type')


args = parser.parse_args()

text = args.text#'majornews'

import tushare as ts
#ts.set_token('') #one time only
ts.set_token('3e7caaeffcf8b35c3419538148106d35990847037d50d07f231d3518')
pro = ts.pro_api()

# 确定起止日期
start_date = '2016-01-01 00:00:00'
end_date = '2019-12-20 23:59:59'

# 做一个用于tushare调用的list
from dateutil import rrule
from datetime import datetime

all_dates_list = []
for dt in rrule.rrule(rrule.DAILY,
                      dtstart=datetime.strptime(start_date.split(' ')[0], '%Y-%m-%d'),
                      until=datetime.strptime(end_date.split(' ')[0], '%Y-%m-%d')):
    all_dates_list.append(dt.strftime('%Y%m%d'))

if text == 'majornews':
    for i in tqdm(all_dates_list):
        if not os.path.exists('data/news/majornews/'+i+'.csv'):
            try:
                df = pro.major_news(src='', start_date=i+' 00:00:00', \
                            end_date=i+' 23:59:59', fields='title,content')
                df.to_csv('data/news/majornews/'+i+'.csv')
                time.sleep(60) #每分钟最多访问该接口2次
            except:
                print ('bad')
                
if text == 'news':
    newssrc = ['sina','wallstreetcn','10jqka','eastmoney','yuncaijing']
    for i in tqdm(range(len(all_dates_list))):
        for src in newssrc:
            if not os.path.exists('data/news/news/'+all_dates_list[i]+src+'.csv'):
                try:
                    df = pro.news(src=src, start_date=all_dates_list[i], \
                                end_date=all_dates_list[i+1])
                    df.to_csv('data/news/news/'+all_dates_list[i]+src+'.csv')
                    time.sleep(7) #每分钟最多访问该接口10次
                except:
                    print ('bad')

if text == 'twitternews':
    for i in tqdm(all_dates_list):
        if not os.path.exists('data/news/twitternews/'+i+'.csv'):
            try:
                df = pro.exchange_twitter(start_date=i+' 00:00:00', end_date=i+' 23:59:59', \
                            fields="id,account,nickname,content,retweet_content,media,str_posted_at,create_at")
                df.to_csv('data/news/twitternews/'+i+'.csv')
                time.sleep(1) #每分钟最多访问该接口2次
            except:
                print ('bad')
if text == 'allcompany':
    allcompany = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    startdate = all_dates_list[0]
    enddate=all_dates_list[-1]
    for i in tqdm(range(len(allcompany))):
        if not os.path.exists('data/news/company/'+allcompany['ts_code'][i]+'.csv'):
            try:
                singlestockalldf = pro.anns(ts_code=allcompany['ts_code'][i], start_date=startdate, end_date=enddate)
                if singlestockalldf.shape[0] ==50:
                    #print ('new')
                    startdate = np.array(df.ann_date)[-1] #表格最后一天
                    df = pro.anns(ts_code=allcompany['ts_code'][i], start_date=startdate, end_date=enddate)
                    singlestockalldf = pd.concat((singlestockalldf,df),axis=0)
                singlestockalldf.to_csv('data/news/company/'+allcompany['ts_code'][i]+'.csv')
                time.sleep(0) #每分钟最多访问该接口10次
            except:
                print ('bad')