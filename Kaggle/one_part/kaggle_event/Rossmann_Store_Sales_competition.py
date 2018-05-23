
# coding: utf-8

# ## 便利店销量预测
# 这是[便利店销量预测比赛](https://www.kaggle.com/c/rossmann-store-sales)的一个简单尝试参考。<br>
# by [@寒小阳](http://blog.csdn.net/han_xiaoyang)

# <h1 class="page-name">
#     Forecast sales using store, promotion, and competitor data
# </h1>
# 
# 
# <p>Rossmann operates over 3,000 drug stores in 7 European countries. Currently, <br />Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.</p>
# <p><span style="font-size: 1em; line-height: 1.5em;">In their first Kaggle competition, Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany. Reliable sales forecasts enable store managers to create effective staff schedules that increase productivity and motivation. By helping Rossmann create a robust prediction model, you will help store managers stay focused on what’s most important to them: their customers and their teams! </span></p>
# <p><span style="font-size: 1em; line-height: 1.5em;"> <img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/4594/media/rossmann_banner2.png" alt="" height="81" width="640" /><br /></span></p>
# <p><em><span style="font-size: 1em; line-height: 1.5em;">If you are interested in joining Rossmann at their headquarters near Hanover, Germany, please contact Mr. Frank König (Frank.Koenig {at} rossmann.de) Rossmann is currently recruiting data scientists at <a href="http://www.rossmann.de/unternehmen/karriere/stellenboerse/stellenanzeigen~jid=3A5205E3-C4F9-4F5D-AA93-438D0B064D70~">senior</a> and <a href="http://www.rossmann.de/unternehmen/karriere/stellenboerse/stellenanzeigen~jid=F5142F37-C823-4767-B7CF-21DE3B351D66~">entry-level</a> positions.</span></em></p>
# 

# ### 数据
# <table id="data-files" class="nicetable full roomy align-top border">   
# <thead>
#     <tr>
#         <th colspan="2">File Name</th> 
#         <th>Available Formats</th>         
#     </tr> 
# </thead>
# 
#     <tbody  >
#         <tr>
# 
#             <td class="file-name" colspan="2" rowspan="1">sample_submission.csv</td>
#             <td>
# <a href="/c/rossmann-store-sales/download/sample_submission.csv.zip" name="sample_submission.csv.zip" onclick="window.Intercom(&#39;trackEvent&#39;,&#39;download_compdata&#39;,{&#39;comp_id&#39;: 4594});">.zip (55.25 kb)</a>                    </td>
#         </tr>
# 
#     </tbody>
#     <tbody  >
#         <tr>
# 
#             <td class="file-name" colspan="2" rowspan="1">store.csv</td>
#             <td>
# <a href="/c/rossmann-store-sales/download/store.csv.zip" name="store.csv.zip" onclick="window.Intercom(&#39;trackEvent&#39;,&#39;download_compdata&#39;,{&#39;comp_id&#39;: 4594});">.zip (8.33 kb)</a>                    </td>
#         </tr>
# 
#     </tbody>
#     <tbody  >
#         <tr>
# 
#             <td class="file-name" colspan="2" rowspan="1">test.csv</td>
#             <td>
# <a href="/c/rossmann-store-sales/download/test.csv.zip" name="test.csv.zip" onclick="window.Intercom(&#39;trackEvent&#39;,&#39;download_compdata&#39;,{&#39;comp_id&#39;: 4594});">.zip (143.25 kb)</a>                    </td>
#         </tr>
# 
#     </tbody>
#     <tbody  >
#         <tr>
# 
#             <td class="file-name" colspan="2" rowspan="1">train.csv</td>
#             <td>
# <a href="/c/rossmann-store-sales/download/train.csv.zip" name="train.csv.zip" onclick="window.Intercom(&#39;trackEvent&#39;,&#39;download_compdata&#39;,{&#39;comp_id&#39;: 4594});">.zip (5.66 mb)</a>                    </td>
#         </tr>
# 
#     </tbody>
# </table>

# <p>You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.</p>

# <h3>Files</h3>
# <ul>
# <li><strong>train.csv</strong> - historical data including Sales</li>
# <li><strong>test.csv</strong> - historical data excluding Sales</li>
# <li><strong>sample_submission.csv</strong> - a sample submission file in the correct format</li>
# <li><strong>store.csv</strong> - supplemental information about the stores</li>
# </ul>
# <h3>Data fields</h3>
# <p>Most of the fields are self-explanatory. The following are descriptions for those that aren't.</p>
# <ul>
# <li><strong>Id</strong> - an Id that represents a (Store, Date) duple within the test set</li>
# <li><strong>Store</strong> - a unique Id for each store</li>
# <li><strong>Sales</strong> - the turnover for any given day (this is what you are predicting)</li>
# <li><strong>Customers</strong> - the number of customers on a given day</li>
# <li><strong>Open</strong> - an indicator for whether the store was open: 0 = closed, 1 = open</li>
# <li><strong>StateHoliday</strong> - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None</li>
# <li><strong>SchoolHoliday</strong> - indicates if the (Store, Date) was affected by the closure of public schools</li>
# <li><strong>StoreType</strong> - differentiates between 4 different store models: a, b, c, d</li>
# <li><strong>Assortment</strong> - describes an assortment level: a = basic, b = extra, c = extended</li>
# <li><strong>CompetitionDistance</strong> - distance in meters to the nearest competitor store</li>
# <li><strong>CompetitionOpenSince[Month/Year]</strong> - gives the approximate year and month of the time the nearest competitor was opened</li>
# <li><strong>Promo</strong> - indicates whether a store is running a promo on that day</li>
# <li><strong>Promo2</strong> - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating</li>
# <li><strong>Promo2Since[Year/Week]</strong> - describes the year and calendar week when the store started participating in Promo2</li>
# <li><strong>PromoInterval</strong> - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store</li>
# </ul>

# ### 引入所需的库

# In[1]:


import pandas as pd
import datetime
import csv
import numpy as np
import os
import scipy as sp
import xgboost as xgb
import itertools
import operator
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
from matplotlib import pylab as plt
plot = True

goal = 'Sales'
myid = 'Id'


# ### 定义一些变换和评判准则
# 使用不同的loss function的时候要特别注意这个

# In[3]:


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


# In[4]:


store = pd.read_csv('./data/store.csv')


# In[5]:


store.head()


# In[6]:


train_df = pd.read_csv('./data/train.csv')


# In[7]:


train_df.head()


# In[8]:


test_df = pd.read_csv('./data/test.csv')


# In[9]:


test_df.head()


# ### 加载数据

# In[10]:


def load_data():
    """
        加载数据，设定数值型和非数值型数据
    """
    store = pd.read_csv('./data/store.csv')
    train_org = pd.read_csv('./data/train.csv',dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('./data/test.csv',dtype={'StateHoliday':pd.np.string_})
    train = pd.merge(train_org,store, on='Store', how='left')
    test = pd.merge(test_org,store, on='Store', how='left')
    features = test.columns.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]
    return (train,test,features,features_non_numeric)


# ### 数据与特征处理

# In[11]:


def process_data(train,test,features,features_non_numeric):
    """
        Feature engineering and selection.
    """
    # # FEATURE ENGINEERING
    train = train[train['Sales'] > 0]

    for data in [train,test]:
        # year month day
        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)

        # promo interval "Jan,Apr,Jul,Oct"
        data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
        data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
        data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
        data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
        data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
        data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
        data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
        data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
        data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
        data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
        data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
        data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)

    # # Features set.
    noisy_features = [myid,'Date']
    features = [c for c in features if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
    features.extend(['year','month','day'])
    # Fill NA
    class DataFrameImputer(TransformerMixin):
        # http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
        def __init__(self):
            """Impute missing values.
            Columns of dtype object are imputed with the most frequent value
            in column.
            Columns of other types are imputed with mean of column.
            """
        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0] # mode
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], # mean
                index=X.columns)
            return self
        def transform(self, X, y=None):
            return X.fillna(self.fill)
    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)
    # Pre-processing non-numberic values
    le = LabelEncoder()
    for col in features_non_numeric:
        le.fit(list(train[col])+list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    # LR和神经网络这种模型都对输入数据的幅度极度敏感，请先做归一化操作
    scaler = StandardScaler()
    for col in set(features) - set(features_non_numeric) -       set([]): # TODO: add what not to scale
        scaler.fit(list(train[col])+list(test[col]))
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train,test,features,features_non_numeric)


# ### 训练与分析

# In[12]:


def XGB_native(train,test,features,features_non_numeric):
    depth = 13
    eta = 0.01
    ntrees = 8000
    mcw = 3
    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": mcw,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    print ("Running with params: " + str(params))
    print ("Running with ntrees: " + str(ntrees))
    print ("Running with features: " + str(features))

    # Train model with local split
    tsize = 0.05
    X_train, X_test = cross_validation.train_test_split(train, test_size=tsize)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train[goal] + 1))
    dvalid = xgb.DMatrix(X_test[features], np.log(X_test[goal] + 1))
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[goal].values)
    print(error)

    # Predict and Export
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({myid: test[myid], goal: np.exp(test_probs) - 1})
    if not os.path.exists('result/'):
        os.makedirs('result/')
    submission.to_csv("./result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.csv" % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)) , index=False)
    # Feature importance
    if plot:
      outfile = open('xgb.fmap', 'w')
      i = 0
      for feat in features:
          outfile.write('{0}\t{1}\tq\n'.format(i, feat))
          i = i + 1
      outfile.close()
      importance = gbm.get_fscore(fmap='xgb.fmap')
      importance = sorted(importance.items(), key=operator.itemgetter(1))
      df = pd.DataFrame(importance, columns=['feature', 'fscore'])
      df['fscore'] = df['fscore'] / df['fscore'].sum()
      # Plotitup
      plt.figure()
      df.plot()
      df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
      plt.title('XGBoost Feature Importance')
      plt.xlabel('relative importance')
      plt.gcf().savefig('Feature_Importance_xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.png' % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)))


# In[ ]:


print ("=> 载入数据中...")
train,test,features,features_non_numeric = load_data()
print ("=> 处理数据与特征工程...")
train,test,features,features_non_numeric = process_data(train,test,features,features_non_numeric)
print("=> 使用XGBoost建模...")
XGB_native(train,test,features,features_non_numeric)

