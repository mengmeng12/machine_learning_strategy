'''
based on uqer
four ML models
multi factor, plus sentiment and heat index, and CCTV news sentiment
classification and regression based
'''

universe = set_universe('HS300') 
from datetime import datetime
def StockFactorsGet(trading_day,stock):
    data =  DataAPI.MktStockFactorsOneDayGet(tradeDate=trading_day.strftime('%Y%m%d'),secID=stock,field=['secID','tradeDate']+used_factors,pandas="1")
    data['ticker'] = data['secID'].apply(lambda x: x[0:6])
    data.set_index('ticker',inplace=True)
    #数据标准化
    for f in used_factors:
        if data[f].std() == 0:
            continue
        data[f] = (data[f] - data[f].mean()) / data[f].std()
    return data

from sklearn.preprocessing import StandardScaler

from CAL.PyCAL import *
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC,SVR
from sklearn.cross_validation import train_test_split
import pandas as pd

######################################################################
##############classification based model (better than HS300 or not)
######################################################################

start = '2016-01-01'                       
end = '2019-12-20'                         
benchmark = 'HS300'                        
universe = set_universe('HS300')  
capital_base = 10000000                      
freq = 'd'                                 
refresh_rate = 30                           
used_factors = ['ROA', 'PE', 'LCAP','ROE','Volatility','HBETA' ]
#for uqer 
import pandas as pd
cumulative_weight =1 #for CCTV news weight loop
ratio_df = pd.read_table('cctv_ratio_subtractaveyear.csv',sep=',',index_col=0)


def initialize(account):                   
    pass

def handle_data(account): 
    cumulative_weight =1
    cal = Calendar('China.SSE')
    period = Period('-30B')
    today = account.current_date
    today = Date.fromDateTime(account.current_date)  
    train_day = cal.advanceDate(today, period)
    train_day = train_day.toDateTime()
    #print ('today,train_day',today,train_day)
    
    # 提取系数回归所需数据
    train_data=StockFactorsGet(train_day,universe)
    train_data=train_data.dropna()
    #print ('train_data shape',train_data.shape)
    # 去极值处理
    for f in used_factors:
        for i in range(len(used_factors)):
            if train_data[f].values[i]>=train_data[f].median()+5.2*train_data[f].mad():
                train_data[f].values[i]=train_data[f].values[i]=train_data[f].median()+5.2*train_data[f].mad()
            elif train_data[f].values[i]<=train_data[f].median()+5.2*train_data[f].mad():
                train_data[f].values[i]=train_data[f].values[i]=train_data[f].median()+5.2*train_data[f].mad()
    # 提取收盘价收益率
    ret_data=DataAPI.MktEqudGet(tradeDate=today.strftime('%Y%m%d'),secID=train_data['secID'],field=['tradeDate','secID','closePrice'],pandas="1")
    preprice_data=DataAPI.MktEqudGet(tradeDate=train_day.strftime('%Y%m%d'),secID=train_data['secID'],field=['tradeDate','secID','closePrice'],pandas="1")
    ret_data['ret']=ret_data['closePrice']/preprice_data['closePrice']-1
   
    # 提取指数收盘价，计算收益率
    hs300_data = DataAPI.MktIdxdGet(tradeDate=train_day.strftime('%Y%m%d'),ticker='000300', field='ticker,tradeDate,closeIndex',pandas="1")
    hs300_today_data=DataAPI.MktIdxdGet(tradeDate=today.strftime('%Y%m%d'),ticker='000300', field='ticker,tradeDate,closeIndex',pandas="1")
    hs300_data['ret']=hs300_today_data['closeIndex']/hs300_data['closeIndex']-1
    # 提取今日因子数据
    today_data = StockFactorsGet(today,train_data['secID'])
    today_data=today_data.fillna(0)
    for index in range(len(ret_data['ret'])):
        if ret_data['ret'].values[index] > hs300_data['ret'].values:
            ret_data['ret'].values[index] = 1
        else:
            ret_data['ret'].values[index] = 0
    # Logistic回归过程
    factors=['ROA', 'PE','ROE','Volatility','HBETA']
    x_train = train_data[factors]
    y_train = ret_data['ret']
    x_test = today_data[factors]
    #classifier = LogisticRegression()
    #classifier = SVC(probability=True)
    #classifier = GradientBoostingClassifier()
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    proba=classifier.predict_proba(x_test)
    today_data['predictions']=predictions
    today_data['proba']=proba[:,1]
    new_data=today_data[today_data['predictions']>0]
    new_data.set_index('secID',inplace=True)
    buy=DataFrame(index=new_data.index, columns=['proba'], data=0)
    buy['proba']=new_data['proba']
    buy=buy.sort(columns=['proba'],ascending=False)
    # 生成列表
    locationin_cctv_df = np.where(ratio_df.index ==int(today.strftime('%Y%m%d')))[0][0]
    average_cctv = np.average(ratio_df.iloc[max(locationin_cctv_df-refresh_rate,0):\
                  max(locationin_cctv_df-refresh_rate,0)+refresh_rate])
    cumulative_weight +=average_cctv
    if cumulative_weight >=1:
        cumulative_weight= 1
    print (cumulative_weight)
    buylist=buy.index.tolist()[:20]
    for stk in account.valid_secpos:
        if stk not in buylist:
            account.order_to(stk, 0)
    # 等权重买入所选股票
    portfolio_value = account.referencePortfolioValue  #参考投资策略价        
    for stk in buylist:
        if np.isnan(account.referencePrice[stk]) or account.referencePrice[stk] == 0:  # 停牌或是还没有上市等原因不能交易
            continue
        if stk not in account.valid_secpos:
            account.order(stk, cumulative_weight*account.referencePortfolioValue / len(buylist) / account.referencePrice[stk])  

# merge sentiment and heat index together
def get_all_data(train_day,universe):
    data =  DataAPI.MktStockFactorsOneDayGet(tradeDate=train_day.strftime('%Y%m%d'),secID=universe,field=['secID','tradeDate']+used_factors,pandas="1")
    data['ticker'] = data['secID'].apply(lambda x: x[0:6])
    data.set_index('ticker',inplace=True)
    heat = DataAPI.NewsHeatIndexV2Get(secID=universe, beginDate=train_day.strftime("%Y%m%d"),endDate=train_day.strftime("%Y%m%d")).iloc[:,np.array([0,6])]
    senti = DataAPI.NewsSentimentIndexV2Get(secID=universe, beginDate=train_day.strftime("%Y%m%d"),endDate=train_day.strftime("%Y%m%d")).iloc[:,np.array([0,6])]
    senti['ticker'] = senti['secID'].apply(lambda x: x[0:6])
    heat['ticker'] = heat['secID'].apply(lambda x: x[0:6])
    senti.set_index('ticker',inplace=True)
    heat.set_index('ticker',inplace=True)
    heat = heat.iloc[:,1]
    senti = senti.iloc[:,1]
    data_ = pd.concat((data,senti.loc[data.index],heat.loc[data.index]),axis=1)
    #数据标准化
    for f in data_.columns[2:]:
        if data_[f].std() == 0:
            continue
        data_[f] = (data_[f] - data_[f].mean()) / data_[f].std()
    return data_

######################################################################
##############regression based model (best top20/10 ret)
######################################################################

start = '2016-01-01'                       
end = '2019-12-20'                         
benchmark = 'HS300'                        
universe = set_universe('HS300')  
capital_base = 10000000                      
freq = 'd'                                 
refresh_rate = 30                           
used_factors = ['ROA', 'PE', 'LCAP','ROE','Volatility','HBETA' ]
#for uqer 
import pandas as pd
cumulative_weight =1 #for CCTV news weight loop
ratio_df = pd.read_table('cctv_ratio_subtractaveyear.csv',sep=',',index_col=0)


def initialize(account):                   
    pass

def handle_data(account): 
    cumulative_weight =1
    cal = Calendar('China.SSE')
    period = Period('-30B')
    today = account.current_date
    today = Date.fromDateTime(account.current_date)  
    train_day = cal.advanceDate(today, period)
    train_day = train_day.toDateTime()
    #print ('today,train_day',today,train_day)
    
    # 提取系数回归所需数据
   
    
    #train_data=StockFactorsGet(train_day,universe)
    train_data=get_all_data(train_day,universe)
    train_data=train_data.dropna()
    #print ('train_data shape',train_data.shape)
    # 去极值处理
    for f in used_factors:
        for i in range(len(used_factors)):
            if train_data[f].values[i]>=train_data[f].median()+5.2*train_data[f].mad():
                train_data[f].values[i]=train_data[f].values[i]=train_data[f].median()+5.2*train_data[f].mad()
            elif train_data[f].values[i]<=train_data[f].median()+5.2*train_data[f].mad():
                train_data[f].values[i]=train_data[f].values[i]=train_data[f].median()+5.2*train_data[f].mad()
    # 提取收盘价收益率
    ret_data=DataAPI.MktEqudGet(tradeDate=today.strftime('%Y%m%d'),secID=train_data['secID'],field=['tradeDate','secID','closePrice'],pandas="1")
    preprice_data=DataAPI.MktEqudGet(tradeDate=train_day.strftime('%Y%m%d'),secID=train_data['secID'],field=['tradeDate','secID','closePrice'],pandas="1")
    ret_data['ret']=ret_data['closePrice']/preprice_data['closePrice']-1
   
    # 提取指数收盘价，计算收益率
    hs300_data = DataAPI.MktIdxdGet(tradeDate=train_day.strftime('%Y%m%d'),ticker='000300', field='ticker,tradeDate,closeIndex',pandas="1")
    hs300_today_data=DataAPI.MktIdxdGet(tradeDate=today.strftime('%Y%m%d'),ticker='000300', field='ticker,tradeDate,closeIndex',pandas="1")
    hs300_data['ret']=hs300_today_data['closeIndex']/hs300_data['closeIndex']-1
    # 提取今日因子数据
    today_data = get_all_data(today,train_data['secID'])
    today_data=today_data.fillna(0)
    #for index in range(len(ret_data['ret'])):
       # if ret_data['ret'].values[index] > hs300_data['ret'].values:
        #    ret_data['ret'].values[index] = 1
       # else:
          #  ret_data['ret'].values[index] = 0
    # 回归
    factors=['ROA', 'PE','ROE','Volatility','HBETA','sentimentIndex','heatIndex']
    x_train = train_data[factors]
    y_train = ret_data['ret']
    x_test = today_data[factors]
    #print (y_train)
    #classifier = LogisticRegression()
    classifier = SVR(kernel='linear')
    #classifier = GradientBoostingRegressor()
    #classifier = RandomForestRegressor()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    #proba=classifier.predict_proba(x_test)
    #print (classifier.feature_importances_)
    weightlist = np.loadtxt('svm_regressor_10_weight.txt',dtype='float')
    print (classifier.coef_[0])
    np.savetxt('svm_regressor_10_weight.txt',\
            # np.concatenate((weightlist,np.array(classifier.feature_importances_)\
               np.concatenate((weightlist,np.array(classifier.coef_[0])\
                            ),axis=0))
    
    today_data['predictions']=predictions
    #today_data['proba']=proba[:,1]
    new_data=today_data#[today_data['predictions']>0]
    new_data.set_index('secID',inplace=True)
    #buy=DataFrame(index=new_data.index, columns=['proba'], data=0)
    #buy['proba']=new_data['proba']
    #buy=buy.sort(columns=['proba'],ascending=False)
    buy=DataFrame(index=new_data.index, columns=['predictions'], data=0)
    #print(buy)
    buy['predictions']=new_data['predictions']
    buy=buy.sort(columns=['predictions'],ascending=False)
    #print(buy)
    # 生成列表
    locationin_cctv_df = np.where(ratio_df.index ==int(today.strftime('%Y%m%d')))[0][0]
    average_cctv = np.average(ratio_df.iloc[max(locationin_cctv_df-refresh_rate,0):\
                  max(locationin_cctv_df-refresh_rate,0)+refresh_rate])
    cumulative_weight +=average_cctv
    if cumulative_weight >=1:
        cumulative_weight= 1
    print (cumulative_weight)
    buylist=buy.index.tolist()[:10]
    for stk in account.valid_secpos:
        if stk not in buylist:
            account.order_to(stk, 0)
    # 等权重买入所选股票
    portfolio_value = account.referencePortfolioValue  #参考投资策略价        
    for stk in buylist:
        if np.isnan(account.referencePrice[stk]) or account.referencePrice[stk] == 0:  # 停牌或是还没有上市等原因不能交易
            continue
        if stk not in account.valid_secpos:
            account.order(stk, cumulative_weight*account.referencePortfolioValue / len(buylist) / account.referencePrice[stk])  