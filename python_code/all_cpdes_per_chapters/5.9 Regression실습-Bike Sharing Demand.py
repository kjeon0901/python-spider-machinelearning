#!/usr/bin/env python
# coding: utf-8

# ## 5.9 Regression 실습 - Bike Sharing Demand
# ### 데이터 클렌징 및 가공

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

bike_df = pd.read_csv('C:/jeon/bike_train.csv') # 여러 날씨 데이터 정보에 기반한 1시간 간격 동안의 자전거 대여 횟수
print(bike_df.shape)
bike_df.head(3)


# datetime: hourly date + timestamp  
# season: 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울  
# holiday: 1 = 주말을 제외한 국경일 등의 휴일, 0 = 휴일이 아닌 날  
# workingday: 1 = 주말 및 휴일이 아닌 주중, 0 = 주말 및 휴일  
# weather:  
# • 1 = 맑음, 약간 구름 낀 흐림  
# • 2 = 안개, 안개 + 흐림  
# • 3 = 가벼운 눈, 가벼운 비 + 천둥  
# • 4 = 심한 눈/비, 천둥/번개  
# temp: 온도(섭씨)   
# atemp: 체감온도(섭씨)  
# humidity: 상대습도  
# windspeed: 풍속  
# casual: 사전에 등록되지 않는 사용자가 대여한 횟수  
# registered: 사전에 등록된 사용자가 대여한 횟수  
# count: 대여 횟수  

# In[2]:


bike_df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    10886 non-null  object  # 얘만 string
 1   season      10886 non-null  int64  
 2   holiday     10886 non-null  int64  
 3   workingday  10886 non-null  int64  
 4   weather     10886 non-null  int64  
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64  
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64  
 10  registered  10886 non-null  int64  
 11  count       10886 non-null  int64  
dtypes: float64(3), int64(8), object(1)
'''


# In[3]:


# 문자열을 datetime 타입으로 변경. 
bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime) # bike_df.datetime == bike_df['datetime']
    # pd.to_datetime를 써서 → datetime type(pandas, 원하는 날짜와 시간만 뽑아 사용 가능)으로 바꿈. 
bike_df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype         
---  ------      --------------  -----         
 0   datetime    10886 non-null  datetime64[ns]   # 이렇게 바뀜!
 1   season      10886 non-null  int64         
 2   holiday     10886 non-null  int64         
 3   workingday  10886 non-null  int64         
 4   weather     10886 non-null  int64         
 5   temp        10886 non-null  float64       
 6   atemp       10886 non-null  float64       
 7   humidity    10886 non-null  int64         
 8   windspeed   10886 non-null  float64       
 9   casual      10886 non-null  int64         
 10  registered  10886 non-null  int64         
 11  count       10886 non-null  int64         
dtypes: datetime64[ns](1), float64(3), int64(8)
'''


# In[4]:


# datetime 타입에서 년, 월, 일, 시간 추출
bike_df['year'] = bike_df.datetime.apply(lambda x : x.year) # bike_df 데이터프레임 안에서 datetime 컬럼의 요소들을 순차적으로 x에 넣어주고, 
                                                            # 그때 x.year를 새로 만든 bike_df['year'] 컬럼에 요소로 하나씩 넣어준다. 
                                                                # -> datetime 으로 타입 캐스팅 해줘서 굳이 인덱스 슬라이싱 안 해주고 편함!!
bike_df['month'] = bike_df.datetime.apply(lambda x : x.month)
bike_df['day'] = bike_df.datetime.apply(lambda x : x.day)
bike_df['hour'] = bike_df.datetime.apply(lambda x: x.hour)
print(bike_df.info()) # 'year', 'month', 'day', 'hour' 4개의 컬럼 추가로 만들어짐. 
bike_df.head(3)


# In[5]:


drop_columns = ['datetime','casual','registered'] # 'datetime'-따로 담아줬으니까 삭제, 'casual','registered'-둘이 합친 게 count로 들어가 있고 딱히 필요없어 보이니까 삭제
bike_df.drop(drop_columns, axis=1,inplace=True)


# ### 로그 변환, 피처 인코딩, 모델 학습/예측/평가 

# In[6]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# log 값 변환 시 언더플로우 영향으로 log() 가 아닌 log1p() 를 이용하여 RMSLE 계산
def rmsle(y, pred): # log1p된 오류 값에 해당하는 rmse : 결괏값, 예측값에 각각 log를 씌움(loga-logb != log(a-b))
    log_y = np.log1p(y) # y 로그캐스팅
    log_pred = np.log1p(pred) # y^ 로그캐스팅
    squared_error = (log_y - log_pred) ** 2 # error == logy-logy^
    rmsle = np.sqrt(np.mean(squared_error)) # log1p된 오류 값에 해당하는 rmse
    return rmsle

# 사이킷런의 mean_square_error() 를 이용하여 RMSE 계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred)) # sqrt: 루트. mean_squared_error(y, y^)로 MSE 구하고 루트씌움. 

# RMSLE, RMSE, MAE 를 모두 계산 
def evaluate_regr(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    # MAE 는 scikit learn의 mean_absolute_error() 로 계산
    mae_val = mean_absolute_error(y,pred)
    print('RMSLE: {0:.3f}, RMSE: {1:.3F}, MAE: {2:.3F}'.format(rmsle_val, rmse_val, mae_val))


# In[7]:


from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression , Ridge , Lasso

y_target = bike_df['count']
X_features = bike_df.drop(['count'],axis=1,inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test ,pred)
'''
RMSLE: 1.165, RMSE: 140.900, MAE: 105.924

MAE와 MSE를 바로 비교하면 절댓값과 제곱을 비교하는 것이기 때문에, 
지금처럼 단순 직접 비교로는 MAE와 RMSE를 비교하는 것이 맞다. 
'''

####################여기까지만 씀!

# In[8]:


def get_top_error_data(y_test, pred, n_tops = 5):
    # DataFrame에 컬럼들로 실제 대여횟수(count)와 예측 값을 서로 비교 할 수 있도록 생성. 
    result_df = pd.DataFrame(y_test.values, columns=['real_count']) # 실제값 y 담은 column 만듦
    result_df['predicted_count']= np.round(pred) # 예측값 y^ 담은 column 만듦
    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count']) # |y-y^|
    # 예측값과 실제값이 가장 큰 데이터 순으로 출력. 
    print(result_df.sort_values('diff', ascending=False)[:n_tops]) # 오류값 가장 큰 순(ascending=False:내림차순)으로 10개만 뽑아봄
    
get_top_error_data(y_test,pred,n_tops=10) # 초기값 5 있지만, 10 넣어줬으니 10이 우선. 


# In[9]:


y_target.hist() # y_target(=='count'컬럼)이 정규분포 이루는지 확인. 
plt.plot(range(0,100), y_target[0:100])
plt.scatter(range(0,100), y_target[0:100])

# In[10]:


y_log_transform = np.log1p(y_target)
y_log_transform.hist()
plt.plot(range(0,100), y_log_transform[0:100])
plt.scatter(range(0,100), y_log_transform[0:100])
'''
skewness(비대칭도)가 높다 == 데이터 분포가 한쪽으로 치우쳐있다
    → 이런 값들을 정규 분포로 바꾸는 방법 : 로그변환★
    
로그변환
- 스케일이 변한 것이지, 분포 그래프의 모양이 바뀌는 게 아님.    // a > b 이면 log1p(a) > log1p(b)
        ex_ 데이터분포 그래프를 f(x)라 할 때
                    원본 : f(70) > f(5) > f(25) > f(100)
              log1p(원본): f(70) > f(5) > f(25) > f(100) 그대로! 그냥 기울기가 완만해질 뿐!!      
- 로그변환 이후 히스토그램은 왜 모양 자체가 바뀔까??
        => 로그변환을 하면 f(x)값의 극단적인 비대칭성이 줄어들고, 새로 그리는 그래프는 step을 그 줄어든 비율만큼 짧게 잡을 것이다. 
           그러면 새로운 도수 분포는 f(x)값이 보다 고르게 퍼지고, 결국 히스토그램이 정규분포를 닮아간다. 
'''
#y_log_log_transform = np.log1p(y_log_transform)
#y_log_log_transform.hist()
#plt.plot(range(0,100), y_log_log_transform[0:100])
#plt.scatter(range(0,100), y_log_log_transform[0:100])

# In[11]:


# 타겟 컬럼인 count 값을 log1p 로 Log 변환
y_target_log = np.log1p(y_target)

# 로그 변환된 y_target_log를 반영하여 학습/테스트 데이터 셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size=0.3, random_state=0) # X_features:피처데이터세트, y_target_log:
                                                                                                             # 로그변환된 아이들로 학습됐으므로, 그거에 맞춰서 w 학습됨. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

# 테스트 데이터 셋의 Target 값은 Log 변환되었으므로 다시 expm1를 이용하여 원래 scale로 변환
y_test_exp = np.expm1(y_test)

# 예측 값 역시 Log 변환된 타겟 기반으로 학습되어 예측되었으므로 다시 exmpl으로 scale변환
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp ,pred_exp)
'''
RMSLE: 1.017, RMSE: 162.594, MAE: 109.286

RMSLE는 줄었지만, RMSE는 오히려 더 늘어남. 
이유 찾아보자!
     ↓↓↓
'''


# In[12]:


coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)
'''
Year 피처의  회귀 계수 값이 독보적으로 크다. 
자전거 대여 횟수에 별 영향 없는 무의미한 값인데, 회귀에서는 숫자가 가중치 등에 큰 영향을 줌. 게다가 값이 2011, 2012로 엄청 큰 값이다. 
=> 원-핫 인코딩 해주자!

원-핫 인코딩 : 유니크한 요소의 개수만큼 컬럼 만들어서 각각 해당되면 1, 아니면 0 집어넣음. 
ex_ 2011, 2012 컬럼 추가해서 각 컬럼에 요소마다 0, 1 집어넣음
    year(2col), month(12col), day(31col), hour(24col) - 이렇게 한다면 총 69개의 컬럼 추가해서 원-핫 인코딩
    컬럼이 너무 늘어나긴 하지만,,,,ㅜㅜ
'''

# In[13]:


# 'year','month','hour','season','weather' feature들을 One Hot Encoding
X_features_ohe = pd.get_dummies(X_features, columns=['year','month','day','hour', 'holiday', 'workingday','season','weather']) # 새로운 df에 원-핫 인코딩 결과 컬럼 추가
    # 참고로, day 20부터는 bike_test.csv 파일에 있음. 
    # 지금은 bike_train.csv 파일 안에서만 실행하고 코드 다룰 것이기 때문에, bike_test.csv와 합칠 걱정 하지 않고 그냥 마음대로 원-핫 인코딩 해줬음. 

# In[14]:


# 원-핫 인코딩이 적용된 feature 데이터 세트 기반으로 다시 학습/예측 데이터 분할. 
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.3, random_state=0)

# 모델과 학습/테스트 데이터 셋을 입력하면 성능 평가 수치를 반환
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train) # 로그변환된 아이들로 학습됐으므로, 그거에 맞춰서 w 학습됨.
    pred = model.predict(X_test) # 예측도 로그 변환 영향 받음. 
    if is_expm1 :
        y_test = np.expm1(y_test) # 로그함수 log(x)의 반대 지수함수 exp(x), log1p(x)의 반대 expm1(x)
        pred = np.expm1(pred)
    print('###',model.__class__.__name__,'###')
    evaluate_regr(y_test, pred)
# end of function get_model_predict    

# model 별로 평가 수행
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)

for model in [lr_reg, ridge_reg, lasso_reg]:
    get_model_predict(model,X_train, X_test, y_train, y_test,is_expm1=True)
'''
### LinearRegression ###
RMSLE: 0.590, RMSE: 97.688, MAE: 63.382
### Ridge ###
RMSLE: 0.590, RMSE: 98.529, MAE: 63.893
### Lasso ###
RMSLE: 0.635, RMSE: 113.219, MAE: 72.803
'''


# In[15]:


coef = pd.Series(lr_reg.coef_ , index=X_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:10]
sns.barplot(x=coef_sort.values , y=coef_sort.index)
'''
이제는 좀 영향력 있는 컬럼 순으로 나옴. 
month_9 - month_8 - month_7 - month_5 - month_6 - month_4 - workingday_0 - workingday_1 - month_10 - month_11 - ....
'''


# In[16]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# 랜덤 포레스트, GBM, XGBoost, LightGBM model 별로 평가 수행
rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)
xgb_reg = XGBRegressor(n_estimators=500)
lgbm_reg = LGBMRegressor(n_estimators=500)

for model in [rf_reg, gbm_reg, xgb_reg, lgbm_reg]:
    get_model_predict(model,X_train, X_test, y_train, y_test,is_expm1=True)
'''
### RandomForestRegressor ###
RMSLE: 0.355, RMSE: 50.321, MAE: 31.134
### GradientBoostingRegressor ###
RMSLE: 0.330, RMSE: 53.349, MAE: 32.744
### XGBRegressor ###
RMSLE: 0.342, RMSE: 51.732, MAE: 31.251
### LGBMRegressor ###
RMSLE: 0.319, RMSE: 47.215, MAE: 29.029

아까 LinearRegression, Ridge, lasso 로 평가한 것보다 훨~씬 결과 좋음!! ^-^ 확실히 최신에 나온 게 퍼포먼스가 좋음~~
'''


# In[ ]:




