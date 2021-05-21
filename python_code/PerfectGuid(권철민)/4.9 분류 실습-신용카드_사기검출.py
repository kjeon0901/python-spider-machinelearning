#!/usr/bin/env python
# coding: utf-8

# ### 데이터 일차 가공 및 모델 학습/예측/평가
# 
# ** 데이터 로드 **

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

card_df = pd.read_csv('C:/jeon/creditcard.csv')
card_df.head(3)

print(card_df.info()) #null값 없음. 'Class' column만 int, 나머지는 float
test = card_df.describe() #'Class' column 보니까 값이 0과 1 뿐일 확률이 있구나!
print(card_df['Class'].value_counts()) #그게 맞다! +극단적으로 1에 해당하는 데이터가 적구나(사기case니까 당연)

# In[2]:


print(card_df.shape)


# ** 원본 DataFrame은 유지하고 데이터 가공을 위한 DataFrame을 복사하여 반환 **

# In[3]:


from sklearn.model_selection import train_test_split

# 인자로 입력받은 DataFrame을 복사 한 뒤 Time 컬럼만 삭제하고 복사된 DataFrame 반환
def get_preprocessed_df(df=None):
    df_copy = df.copy() #복사본
    df_copy.drop('Time', axis=1, inplace=True) #정확한 어떤 정보인지는 고객정보니까 알려주지 않았지만, 그냥 계속 증가하는데 별 의미 없어 보임. 그래서 걍 삭제.
    return df_copy


# ** 학습과 테스트 데이터 세트를 반환하는 함수 생성. 사전 데이터 처리가 끝난 뒤 해당 함수 호출  **

# In[4]:


# 사전 데이터 가공 후 학습과 테스트 데이터 세트를 반환하는 함수.
def get_train_test_dataset(df=None):
    # 인자로 입력된 DataFrame의 사전 데이터 가공이 완료된 복사 DataFrame 반환
    df_copy = get_preprocessed_df(df)
    
    # DataFrame의 맨 마지막 컬럼이 레이블, 나머지는 피처들
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]
    
    # train_test_split( )으로 학습과 테스트 데이터 분할. stratify=y_target으로 Stratified 기반 분할
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    
    # 학습과 테스트 데이터 세트 반환
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)


# In[5]:


print('학습 데이터 레이블 값 비율')
print(y_train.value_counts()/y_train.shape[0] * 100)
print('테스트 데이터 레이블 값 비율')
print(y_test.value_counts()/y_test.shape[0] * 100)


# In[6]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

# 수정된 get_clf_eval() 함수 
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    '''
    [[85282    13]
     [   56    92]]
    정확도: 0.9992, 정밀도: 0.8762, 재현율: 0.6216, F1: 0.7273, AUC:0.9582
    카드사기이기 때문에 재현율이 낮으면 안됨. FN(사기가 Positive인데 Negative라고 잘못 예측하면 x!!)
    '''
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))


# In[7]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)

lr_pred = lr_clf.predict(X_test)
lr_pred_proba = lr_clf.predict_proba(X_test)[:, 1]

# 3장에서 사용한 get_clf_eval() 함수를 이용하여 평가 수행. 
get_clf_eval(y_test, lr_pred, lr_pred_proba)


# ** 앞으로 피처 엔지니어링을 수행할 때마다 모델을 학습/예측/평가하므로 이를 위한 함수 생성 ** 

# In[8]:


# 인자로 사이킷런의 Estimator객체와, 학습/테스트 데이터 세트를 입력 받아서 학습/예측/평가 수행.
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:, 1]
    get_clf_eval(tgt_test, pred, pred_proba)
    '''
    아래에서 LightGbm 써서 오차행렬 보여주니까
    [[85290     5]
     [   36   112]]
    정확도: 0.9995, 정밀도: 0.9573, 재현율: 0.7568, F1: 0.8453, AUC:0.9790
    재현율 확 올라감. 게다가 정밀도까지 올라감. 
    '''


# ** LightGBM 학습/예측/평가.**
# 
# (boost_from_average가 True일 경우 레이블 값이 극도로 불균형 분포를 이루는 경우 재현률 및 ROC-AUC 성능이 매우 저하됨.)  
#    LightGBM 2.1.0 이상 버전에서 이와 같은 현상 발생 

# In[9]:


from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False) #n_jobs=-1: estimator 돌릴 때 모든 cpu 사용해서 한번에 돌려주세요~ 컴퓨터는 느려지지만 속도 빨라짐!
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)


# ### 중요 데이터 분포도 변환 후 모델 학습/예측/평가
# 

# ** 중요 feature의 분포도 확인 **

# In[10]:


import seaborn as sns

plt.figure(figsize=(8, 4))
plt.xticks(range(0, 30000, 1000), rotation=60) #rotation=60 : x축 좌표 표기들을 60도 각도 틀어서 써줌
sns.distplot(card_df['Amount']) #Amount : 신용카드 결제 금액. 결제 금액 별 금융사기가 발생한 히스토그램(distplot:몇번 발생했는지 도수분포)
    #kde=True로 default. --> 히스토그램의 밀도 추세 곡선(연속된 값으로) 그려줌. 얘는 금액이기 때문에 이산적인 값이 아니라 연속된 값으로 봐야 함. 
    #이산적인 값 : 확률질량함수
    #연속된 값 : 적분!(dx : x를 굉장히 작게 자른 한 부분) -> 확률밀도함수
# ** 데이터 사전 가공을 위한 별도의 함수에 StandardScaler를 이용하여 Amount 피처 변환 **

# In[11]:


from sklearn.preprocessing import StandardScaler

# 사이킷런의 StandardScaler를 이용하여 정규분포 형태로 Amount 피처값 변환하는 로직으로 수정. 
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1)) #fit_transform() 안에 가로로 된 series(1차원 데이터)가 아닌 세로로 된 2차원 데이터가 들어가주어야 해서 reshape해줌. 
    
    # 변환된 Amount를 Amount_Scaled로 피처명 변경후 DataFrame맨 앞 컬럼으로 입력
    df_copy.insert(0, 'Amount_Scaled', amount_n) #cf. append는 맨 뒤에만 갖다 붙일 수 있다. 
    
    # 기존 Time, Amount 피처 삭제
    df_copy.drop(['Time','Amount'], axis=1, inplace=True) #Time은 애초에 의미 없었고, 기존의 Amount도 필요 없어졌으므로 drop
    return df_copy


# ** StandardScaler 변환 후 로지스틱 회귀 및 LightGBM 학습/예측/평가 **

# In[12]:


# Amount를 정규분포 형태로 변환 후 로지스틱 회귀 및 LightGBM 수행. 
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('### 로지스틱 회귀 예측 성능 ###')
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)


# ** Amount를 로그 변환 **

# In[13]:


def get_preprocessed_df(df=None):
    df_copy = df.copy()
    # 넘파이의 log1p( )를 이용하여 Amount를 로그 변환 
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    return df_copy


# In[14]:


# log1p 와 expm1 설명 
'''
Log Scale
: 데이터 차이가 극단적으로 차이 많이 나는 경우 log 취해주면 한결 보기 편해짐
ex_         x = 1,000,000   10,000      100     10  →  그래프 그리면 1,000,000을 제외하고 모두 다닥다닥 바닥에 붙어버림
       log(x) =     6           4       2       1
log1p
: log(1+x) 사용. log(1)==0이므로 log scale 결괏값이 음수가 나오지 않게 하기 위해 1+x를 넣어줌. 
'''
import numpy as np

print(1e-1000 == 0.0)

print(np.log(1e-1000))

print(np.log(1e-1000 + 1))
print(np.log1p(1e-1000))


# In[15]:


var_1 = np.log1p(100)
var_2 = np.expm1(var_1)
print(var_1, var_2)


# In[16]:


X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('### 로지스틱 회귀 예측 성능 ###')
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)


# ### 이상치 데이터 제거 후 모델 학습/예측/평가

# ** 각 피처들의 상관 관계를 시각화. 결정 레이블인 class 값과 가장 상관도가 높은 피처 추출 **

# In[21]:


import seaborn as sns

plt.figure(figsize=(9, 9))
corr = card_df.corr() 
'''
corr() : 상관계수
column1이 1-2-3-4-5-6 증가하는데 column2도 2-4-6-8-10-12 같이 증가하는 경향성을 보인다면 '양의 상관관계가 있으며 상관계수는 1이다'

완전 똑같이 움직이면                     corr() = 1
완전 반대로 움직이면                     corr() = -1   (어쨌든 얘도 상관관계가 ↑높은↑ 것!!)
방향은 같은데 정도가 다르면 소수점.        corr() = 0.~~
하나는 바뀌는데 하나는 값의 변화가 없다면   corr() = 0
'''
sns.heatmap(corr, cmap='RdBu') #
'''
시각화. 각 피처별로 모든 상관도 관계성을 보여줌. 
RdBu : Red-down Blue-up. 자신과 자신의 corr()값은 1이므로 가장 진한 파란색으로 표현. 

'Class'와 어떤 다른 column이 양의 상관관계/음의 상관관계가 있는 것이 있나?
=> V14, V17이 그나마 음의 상관관계가 진하다!
=> 그럼 얘네한테서 이상치(Outlier) 제거해주는 게 의미가 있겠구나!
'''


# ** Dataframe에서 outlier에 해당하는 데이터를 필터링하기 위한 함수 생성. outlier 레코드의 index를 반환함. **

# In[30]:


import numpy as np

def get_outlier(df=None, column=None, weight=1.5): #이상치(Outlier) 인덱스 리턴 함수
    # fraud에 해당하는 column 데이터만 추출, 1/4 분위와 3/4 분위 지점을 np.percentile로 구함. 
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    
    # IQR을 구하고, IQR에 1.5를 곱하여 최대값과 최소값 지점 구함. 
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    
    # 최대값 보다 크거나, 최소값 보다 작은 값을 boolean index로 접근해서 아웃라이어로 설정하고 걔네의 DataFrame index만을 반환. 
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    
    return outlier_index
'''
[시각화 - 박스 플롯]
이상치     : 최댓값 이상
최댓값     : 3/4 + 1.5*IQR
IQR       : 3/4, Q3(75%)            //IQR = Q1+Q2+Q3. 사분위(Quantile)값의 편차
            2/4, Q2(50%)
            1/4, Q1(25%)
최솟값     : 1/4 - 1.5*IQR
이상치     : 최솟값 이하
'''


# In[28]:


#np.percentile(card_df['V14'].values, 100)
np.max(card_df['V14'])


# In[31]:


outlier_index = get_outlier(df=card_df, column='V14', weight=1.5)
print('이상치 데이터 인덱스:', outlier_index)


# **로그 변환 후 V14 피처의 이상치 데이터를 삭제한 뒤 모델들을 재 학습/예측/평가**

# In[32]:


# get_processed_df( )를 로그 변환 후 V14 피처의 이상치 데이터를 삭제하는 로직으로 변경. 
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n) #아까랑 똑같이 Amount_Scaled
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    
    # 이상치 데이터 삭제하는 로직 추가
    outlier_index = get_outlier(df=df_copy, column='V14', weight=1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True) #axis=0 : 로우방향. 로우를 날려줌. 
    return df_copy

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('### 로지스틱 회귀 예측 성능 ###')
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)
'''
[[85290     5]
 [   25   121]]
정확도: 0.9996, 정밀도: 0.9603, 재현율: 0.8288, F1: 0.8897, AUC:0.9780
=> 재현율: 아까 최대로 올린 0.7635에서 0.8288까지 확 올라옴! FN이 확 줄었구나. 

근데 outlier 삭제해서 더 안 좋아지는 경우도 있음. 모든 게 다 케바케. 
'''

# ### SMOTE 오버 샘플링 적용 후 모델 학습/예측/평가

# In[33]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())


# In[34]:


y_train.value_counts()


# ** 로지스틱 회귀로 학습/예측/평가 **

# In[35]:


lr_clf = LogisticRegression()
# ftr_train과 tgt_train 인자값이 SMOTE 증식된 X_train_over와 y_train_over로 변경됨에 유의
get_model_train_eval(lr_clf, ftr_train=X_train_over, ftr_test=X_test, tgt_train=y_train_over, tgt_test=y_test)


# ** Precision-Recall 곡선 시각화 **

# In[36]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve
get_ipython().run_line_magic('matplotlib', 'inline')

def precision_recall_curve_plot(y_test , pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    


# In[37]:


precision_recall_curve_plot( y_test, lr_clf.predict_proba(X_test)[:, 1] )


# ** LightGBM 모델 적용 **

# In[38]:


lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)
get_model_train_eval(lgbm_clf, ftr_train=X_train_over, ftr_test=X_test,
                  tgt_train=y_train_over, tgt_test=y_test)


# In[ ]:




