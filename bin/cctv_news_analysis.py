import gc, argparse, sys, os, errno
%pylab inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
#sns.set_style('whitegrid')
import h5py
from PIL import Image
import os
from tqdm import tqdm_notebook as tqdm
import scipy
import sklearn
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import jieba
from collections import Counter
import jieba.analyse
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显
plt.rc('axes',axisbelow=True)

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

cctv_all = pd.read_table('data/news/cctv/20160101.csv',sep=',',index_col=0)
for i in tqdm(all_dates_list[1:]):
    cctv_all = pd.concat((cctv_all,pd.read_table('data/news/cctv/'+i+'.csv',sep=',',index_col=0)))
cctv_all.to_csv('data/news/cctv/all_cctv.csv')

#过滤关键词
blacklist = ['责任编辑', '一定','一年', '一起', '一项', '一点儿', '一度','一系列','一道','一次','一亿','进行', '实现', '已经', '指出',
            '为什么', '是不是', '”', '一个', '一些', 'cctv', '一边', '一部', '一致', '一窗', '万亿元', '亿元', '一致同意', '本台记住', '发生', 
            '上述', '不仅', '不再 ', '下去', '首次', '合作', '发展', '国家', '加强', '共同', '重要', '我们', '你们', '他们', '目前',
            '领导人', '推进', '中方', '坚持', '支持', '表示', '时间', '协调', '时间', '制度', '工作', '强调', '进行', '推动', '通过',
            '北京时间', '有没有', '新闻联播', '本台消息', '这个', '那个', '就是', '今天', '明天', '参加', '今年', '明天']

#新增关键词
stopwords = ['一带一路', '雄安新区', '区块链', '数字货币', '虚拟货币',  '比特币', '对冲基金', '自贸区', '自由贸易区','乡村振兴','美丽中国','共享经济','租购同权','新零售',
             '共有产权房','楼市调控', '产权保护', '互联网金融', '5G', '4G', '国企改革', '大湾区', '长江经济带']

for word in stopwords:
        jieba.add_word(word)

mylist = []
cctv_all_df = cctv_all[cctv_all.content.isnull() == False]

pickdate = all_dates_list[-20]
useddf = cctv_all_df[cctv_all_df['date'] ==int(pickdate)]
mylist = list(useddf.title.values.astype('str')) #decode should be string
word_list = []
#对标题内容进行分词（即切割为一个个关键词）
word_list = [" ".join(jieba.cut(sentence)) for sentence in mylist]
new_text = ' '.join(word_list)
#for sentence in tqdm(mylist):
 #   word = jieba.cut(sentence)
#    word_list.append(word)

img = np.array(Image.open("data/chinamap.png"))
img[img>0]=255

import PIL
img = Image.open("data/chinamap.png")
img = np.array(img.resize((800,800), PIL.Image.ANTIALIAS))
img[img>0]=255


def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(100.0 * float(random_state.randint(20, 100)) / 255.0)
    s = int(100.0 * float(random_state.randint(20, 100)) / 255.0)
    l = int(100.0 * float(random_state.randint(20, 100)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)
def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(230,100%%, %d%%)" % np.random.randint(20,80))


def draw_wc(ind):
    pickdate = all_dates_list[ind]
    useddf = cctv_all_df[cctv_all_df['date'] ==int(pickdate)]
    mylist = list(useddf.title.values.astype('str')) #decode should be string
    word_list = []
    #对标题内容进行分词（即切割为一个个关键词）
    word_list = [" ".join(jieba.cut(sentence)) for sentence in mylist]
    new_text = ' '.join(word_list)
    wc = WordCloud(background_color="white", max_words=100, mask=img,contour_color='blue',
                   stopwords=stopwords, max_font_size=100, random_state=42,
                   width=800, height=800,
                   color_func=grey_color_func,
                  # color_func=random_color_func,
                  font_path="/Users/james/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")


    #plt.figure( figsize=(12,8))
    plt.imshow(wc.generate(new_text),interpolation='bilinear')
    plt.title('CCTV WORD CLOUD\n(%s)' %str(pickdate[:4]+'-'+str(pickdate[4:6])+'-'+str(pickdate[6:])),fontsize=18)
    plt.axis("off")
    #plt.tight_layout(pad=0)
    plt.tight_layout()
    #plt.show()


import matplotlib.animation as animation
from IPython.display import HTML
fig, ax = plt.subplots(figsize=(12,8), dpi=100)
plt.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.1,hspace=0,wspace=0)  
animator = animation.FuncAnimation(fig, draw_wc, frames=np.arange(-100,-1,1),interval=200)
animator.save('wc_animation1.mp4')
HTML(animator.to_jshtml())

######sentiment analysis

path='notebooks/'
dictionary=open(path+'否定词.txt','r',encoding='utf-8')
dict=[]
for word in dictionary:
    word=word.strip('\n')
    dict.append(word)
print(dict)

def word_processing(text):
    with open(path+'{}.txt'.format(text),encoding='utf-8') as f:
        word=[w.strip() for w in f.readlines()]
    return word
def judgeodd(n):
    if (n%2)==0:
        return 'even'
    else:
        return 'odd'
deny_word=word_processing('否定词')
posdict=word_processing('positive')
negdict=word_processing('negative')
degree_word=word_processing('程度级别词语')

mostdict = degree_word[degree_word.index('extreme')+1 : degree_word.index('very')]
#权重4，即在情感词前乘以4
verydict = degree_word[degree_word.index('very')+1 : degree_word.index('more')]
#权重3
moredict = degree_word[degree_word.index('more')+1 : degree_word.index('ish')]
#权重2
ishdict = degree_word[degree_word.index('ish')+1 : degree_word.index('last')]
#权重0.5

def sentiment_score_list(dataset,islist=False):
    if islist:
        seg_sentence = dataset
    else:
        seg_sentence = dataset.split('。')

    count1 = []
    count2 = []
    for sen in seg_sentence: #循环遍历每一个评论
        segtmp = jieba.lcut(sen, cut_all=False)  
        #把句子进行分词，以列表的形式返回
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置
        poscount = 0 #积极词的第一次分值
        poscount2 = 0 #积极词反转后的分值
        poscount3 = 0 #积极词的最后分值（包括叹号的分值）
        negcount = 0
        negcount2 = 0
        negcount3 = 0
        for word in segtmp:
            if word in posdict:  # 判断词语是否是情感词
                poscount += 1
                c = 0
                for w in segtmp[a:i]:  # 扫描情感词前的程度词
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                        poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in deny_word:
                        c += 1
                if judgeodd(c) == 'odd':  # 扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i + 1  # 情感词的位置变化

            elif word in negdict:  # 消极情感的分析，与上面一致
                negcount += 1
                d = 0
                for w in segtmp[a:i]:
                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in degree_word:
                        d += 1
                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0
                a = i + 1
            elif word == '！' or word == '!':  ##判断句子是否有感叹号
                for w2 in segtmp[::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict or negdict:
                        poscount3 += 2
                        negcount3 += 2
                        break
            i += 1 # 扫描词位置前移


            # 以下是防止出现负数的情况
            pos_count = 0
            neg_count = 0
            if poscount3 < 0 and negcount3 > 0:
                neg_count += negcount3 - poscount3
                pos_count = 0
            elif negcount3 < 0 and poscount3 > 0:
                pos_count = poscount3 - negcount3
                neg_count = 0
            elif poscount3 < 0 and negcount3 < 0:
                neg_count = -poscount3
                pos_count = -negcount3
            else:
                pos_count = poscount3
                neg_count = negcount3

            count1.append([pos_count, neg_count])
        count2.append(count1)
        count1 = []

    return count2

def sentiment_score(senti_score_list):
    import numpy as np
    score = []
    for review in senti_score_list:
        score_array = np.array(review)
        Pos = np.sum(score_array[:, 0])
        Neg = np.sum(score_array[:, 1])
        AvgPos = np.mean(score_array[:, 0])
        AvgPos = float('%.1f'%AvgPos)
        AvgNeg = np.mean(score_array[:, 1])
        AvgNeg = float('%.1f'%AvgNeg)
        StdPos = np.std(score_array[:, 0])
        StdPos = float('%.1f'%StdPos)
        StdNeg = np.std(score_array[:, 1])
        StdNeg = float('%.1f'%StdNeg)
        score.append([Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg])
    return score

def pos_neg_num(data):
    p=0
    n=0
    for i in range(len(data)):
        if data[i][0]>data[i][1]:
            p+=1
        else:
            n+=1
    return p,n

def get_cctv_news(ind):
    pickdate = all_dates_list[ind]
    useddf = cctv_all_df[cctv_all_df['date'] ==int(pickdate)]
    mylist = list(useddf.content.values.astype('str')) #decode should be string
    
    return mylist

emotion_num = pd.DataFrame(index=np.arange(len(all_dates_list)),\
                           columns=['positive','negative'])
for i in tqdm(range(len(all_dates_list))):
    df=sentiment_score(sentiment_score_list(get_cctv_news(i),islist=True))
    #print(df)
    p,n=pos_neg_num(df)
    emotion_num.iloc[i] = np.array([p,n])

cctv_ratio = np.array(emotion_num.iloc[:,0]/(emotion_num.iloc[:,0]+emotion_num.iloc[:,1]))

fig,ax=plt.subplots(1,figsize=(15,4))
ax.set_xticks(np.arange(0,len(all_dates_list),50))

ax.plot(all_dates_list,\
        ratio,color='g',linewidth=2,alpha=0.8)
ax.set_title('CCTV Positive News Rate')
plt.setp(plt.gca().xaxis.get_majorticklabels(),
         'rotation', 45)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scalar1= StandardScaler()
scalar2= MinMaxScaler()

index = np.arange(0,len(all_dates_list),3).astype('int')
ratio = cctv_ratio-np.average(cctv_ratio)

colors = cm.Spectral_r(scalar2.fit_transform(ratio.reshape(-1,1)).ravel())[index]
#colors = cm.Spectral_r(ratio)[index]
fig =plt.figure(figsize=(20,4), dpi=100)
plt.xticks(np.arange(0,len(all_dates_list),20),rotation=45)
plt.subplots_adjust(top=1,bottom=0,left=0,right=0.9,hspace=0,wspace=0)
plt.bar(np.array(all_dates_list)[index],ratio[index],color=colors,width=1,align="center",zorder=1)
plt.plot(np.array(all_dates_list)[index],ratio[index], color='k',zorder=1,alpha=0.5)
#plt.scatter(all_dates_list[-1], ratio[-1], color='white',s=150,edgecolor ='k',linewidth=2,zorder=3)
#plt.text(all_dates_list[-1], ratio[-1]*1.05,s=np.round(ratio[-1],1),size=10,ha='center', va='top')
plt.ylim(-0.4, 0.1)
plt.margins(x=0.01)
ax = plt.gca()#获取边框
ax.spines['top'].set_color('none')   # 设置上‘脊梁’为无色
ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
ax.spines['left'].set_color('none')   # 设置上‘脊梁’为无色
plt.grid(axis="y",c=(217/256,217/256,217/256),linewidth=1)   #设置网格线   
plt.title('CCTV Positive News Rate (Subtract Average)',fontsize=20)
plt.show()

index = np.arange(0,len(all_dates_list),15).astype('int')[:-1]

colors = cm.Spectral_r(scalar2.fit_transform(ratio.reshape(-1,1)).ravel())
#colors = cm.Spectral_r(ratio)[index]
fig =plt.figure(figsize=(20,4), dpi=100)
plt.xticks(np.arange(0,np.array(all_dates_list)[index].shape[0],5),rotation=45)
plt.subplots_adjust(top=1,bottom=0,left=0,right=0.9,hspace=0,wspace=0)
plt.bar(np.array(all_dates_list)[index],ratio ,color=colors,width=1,align="center",zorder=1)
plt.plot(np.array(all_dates_list)[index],ratio , color='k',zorder=1,alpha=0.5)
#plt.scatter(all_dates_list[-1], ratio[-1], color='white',s=150,edgecolor ='k',linewidth=2,zorder=3)
#plt.text(all_dates_list[-1], ratio[-1]*1.05,s=np.round(ratio[-1],1),size=10,ha='center', va='top')
plt.ylim(-0.15, 0.1)
plt.margins(x=0.01)
ax = plt.gca()#获取边框
ax.spines['top'].set_color('none')   # 设置上‘脊梁’为无色
ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
ax.spines['left'].set_color('none')   # 设置上‘脊梁’为无色
plt.grid(axis="y",c=(217/256,217/256,217/256),linewidth=1)   #设置网格线   
plt.title('CCTV Positive News Rate Average (Period=15 days)',fontsize=20)
plt.show()

fig,ax=plt.subplots(1,figsize=(8,4), dpi=100)
ax.hist(ratio,bins=50,color='g',alpha=0.8)
ax.set_title('CCTV Positive News Rate Distribution')

aveyear = np.ndarray([4])
for i in range(4):
    aveyear[i] = np.average(cctv_ratio[i*365:(i+1)*365])
aveyear_ = np.concatenate((np.repeat(aveyear[0],366),np.repeat(aveyear[1],365),np.repeat(aveyear[2],365),\
np.repeat(aveyear[3],354)))

ratio = cctv_ratio-aveyear_#np.median(cctv_ratio)
ratio_df = pd.DataFrame(ratio,index=all_dates_list,columns=['ratio'])

ratio_df.to_csv('data/cctv_ratio_subtractaveyear.csv')


cumulative_weight =1
today = datetime(2019,1,4).strftime('%Y%m%d')
refresh_rate = 30
locationin_cctv_df = np.where(ratio_df.index ==today)[0][0]
max(locationin_cctv_df-refresh_rate,0)

cum_weigt = []
for i in np.arange(0,len(all_dates_list),15):
    today = all_dates_list[i]
    locationin_cctv_df = np.where(ratio_df.index ==today)[0][0]
    average_cctv = np.average(ratio_df.iloc[max(locationin_cctv_df-refresh_rate,0):\
                  max(locationin_cctv_df-refresh_rate,0)+refresh_rate])
    cumulative_weight +=average_cctv
    if cumulative_weight >=1:
        cumulative_weight= 1
    #print (cumulative_weight)
    cum_weigt.append(cumulative_weight)

#for uqer 
ratio_df = pd.read_table('data/cctv_ratio_subtractaveyear.csv',sep=',',index_col=0)
locationin_cctv_df = np.where(ratio_df.index ==today.strftime('%Y%m%d'))[0][0]
average_cctv = np.average(ratio_df.iloc[max(locationin_cctv_df-refresh_rate,0):\
              max(locationin_cctv_df-refresh_rate,0)+refresh_rate])
cumulative_weight +=average_cctv
if cumulative_weight >=1:
    cumulative_weight= 1

index = np.arange(0,len(all_dates_list),15).astype('int')
ratio = np.array(cum_weigt)
colors = cm.Spectral_r(scalar2.fit_transform(ratio.reshape(-1,1)).ravel())
#colors = cm.Spectral_r(ratio)[index]
fig =plt.figure(figsize=(20,4), dpi=100)
plt.xticks(np.arange(0,np.array(all_dates_list)[index].shape[0],5),rotation=45)
plt.subplots_adjust(top=1,bottom=0,left=0,right=0.9,hspace=0,wspace=0)
plt.bar(np.array(all_dates_list)[index],ratio ,color=colors,width=1,align="center",zorder=1)
plt.plot(np.array(all_dates_list)[index],ratio , color='k',zorder=1,alpha=0.5)
#plt.scatter(all_dates_list[-1], ratio[-1], color='white',s=150,edgecolor ='k',linewidth=2,zorder=3)
#plt.text(all_dates_list[-1], ratio[-1]*1.05,s=np.round(ratio[-1],1),size=10,ha='center', va='top')
plt.ylim(0.3, 1)
plt.margins(x=0.01)
ax = plt.gca()#获取边框
ax.spines['top'].set_color('none')   # 设置上‘脊梁’为无色
ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
ax.spines['left'].set_color('none')   # 设置上‘脊梁’为无色
plt.grid(axis="y",c=(217/256,217/256,217/256),linewidth=1)   #设置网格线   
plt.title('Weight according to CCTV Positive News Rate Average (Period=15 days)',fontsize=20)
plt.show()

featurename = ['ROA', 'PE','ROE','Volatility','HBETA','Sentiment','Heat']
linestyle = ['solid', 'dotted','dashed',  'dashdot','solid', 'dotted','dashed']

rfweight = np.loadtxt('data/rf_regressor_10_weight.txt').reshape(-1,7)
fig,ax=plt.subplots(figsize=(12,5),dpi=100)
index = np.arange(0,len(all_dates_list),45).astype('int')
plt.xticks(rotation=45)
for i in range(7):
    ax.plot(np.array(all_dates_list)[index],rfweight[:,i],label=featurename[i],
            linewidth=2+i*0.25,linestyle=linestyle[i])
ax.set_title('Random Forest Model Feature Importance',fontsize=20)
ax.legend()

weight_df = pd.DataFrame(rfweight,columns=featurename)
fig,ax=plt.subplots(figsize=(12,5),dpi=100)
weight_df.plot(kind='bar', stacked=True,ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('Random Forest Model Feature Importance',fontsize=20)
ax.set_xticklabels(np.array(all_dates_list)[index],rotation=45)

rfweight = np.loadtxt('data/xgboost_regressor_10_weight.txt').reshape(-1,7)
fig,ax=plt.subplots(figsize=(12,5),dpi=100)
index = np.arange(0,len(all_dates_list),45).astype('int')
plt.xticks(rotation=45)
for i in range(7):
    ax.plot(np.array(all_dates_list)[index],rfweight[:,i],label=featurename[i],
            linewidth=2+i*0.25,linestyle=linestyle[i])
ax.set_title('Gradient Boosting Model Feature Importance',fontsize=20)
ax.legend()

weight_df = pd.DataFrame(rfweight,columns=featurename)
fig,ax=plt.subplots(figsize=(12,5),dpi=100)
weight_df.plot(kind='bar', stacked=True,ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('Gradient Boosting Model Feature Importance',fontsize=20)
ax.set_xticklabels(np.array(all_dates_list)[index],rotation=45)

rfweight = np.loadtxt('data/svm_regressor_10_weight.txt').reshape(-1,7)
fig,ax=plt.subplots(figsize=(12,5),dpi=100)
index = np.arange(0,len(all_dates_list),45).astype('int')
plt.xticks(rotation=45)
for i in range(7):
    ax.plot(np.array(all_dates_list)[index],rfweight[:,i],label=featurename[i],
            linewidth=2+i*0.25,linestyle=linestyle[i])
ax.set_title('SVM Regression Model (linear kernel) Feature Weight',fontsize=20)
ax.legend()

weight_df = pd.DataFrame(rfweight,columns=featurename)
fig,ax=plt.subplots(figsize=(12,5),dpi=100)
weight_df.plot(kind='bar', stacked=True,ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('SVM Regression Model (linear kernel) Feature Weight',fontsize=20)
ax.set_xticklabels(np.array(all_dates_list)[index],rotation=45)

