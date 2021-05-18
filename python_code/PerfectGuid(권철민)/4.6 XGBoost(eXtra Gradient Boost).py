#!/usr/bin/env python
# coding: utf-8

# * XGBoost 버전 확인

# In[1]:


import xgboost

print(xgboost.__version__)


# ### 파이썬 래퍼 XGBoost 적용 – 위스콘신 Breast Cancer 데이터 셋
# 
# ** 데이터 세트 로딩 **

# In[2]:


import xgboost as xgb
from xgboost import plot_importance

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
X_features= dataset.data
y_label = dataset.target

cancer_df = pd.DataFrame(data=X_features, columns=dataset.feature_names)
cancer_df['target']= y_label
cancer_df.head(3)


# In[3]:


print(dataset.target_names)
print(cancer_df['target'].value_counts())


# In[4]:


# 전체 데이터 중 80%는 학습용 데이터, 20%는 테스트용 데이터 추출
X_train, X_test, y_train, y_test=train_test_split(X_features, y_label,
                                         test_size=0.2, random_state=156 )
print(X_train.shape , X_test.shape)


# ** 학습과 예측 데이터 세트를 DMatrix로 변환 **

# In[5]:

#여기부터 이제 생소해짐!
dtrain = xgb.DMatrix(data=X_train , label=y_train) #xgboost의 DMatrix() → variable explorer에서 dtrain, dtest 못까본다. 우선 type은 core.DMatrix이다. 
dtest = xgb.DMatrix(data=X_test , label=y_test) #X_test(피처값), y_test(레이블값(정답))이 합쳐졌기 때문에 검증데이터로 사용 가능하다. 


# ** 하이퍼 파라미터 설정 **

# In[6]:


params = { 'max_depth':3,
           'eta': 0.1,
           'objective':'binary:logistic',
           'eval_metric':'logloss', #검증에 사용되는 함수를 logloss로 줌. 매 step마다 logloss가 얼마나 발생했는지 리턴할 것
           'early_stoppings':100
        }
num_rounds = 400 #부스팅 반복횟수(estimator) 400개..!


# ** 주어진 하이퍼 파라미터와 early stopping 파라미터를 train( ) 함수의 파라미터로 전달하고 학습 **

# In[7]:


# train 데이터 셋은 ‘train’ , evaluation(test) 데이터 셋은 ‘eval’ 로 명기합니다. 
wlist = [(dtrain,'train'),(dtest,'eval') ] #검증데이터 1개, 2개, 3개, ... 개수는 상관 없음. 

# 하이퍼 파라미터와 early stopping 파라미터를 train( ) 함수의 파라미터로 전달
xgb_model = xgb.train(params = params , dtrain=dtrain , num_boost_round=num_rounds , evals=wlist ) #sklearn의 fit()처럼 학습시켜서 모델 던져줌
#원래는 검증데이터, 테스트데이터, 트레인데이터 따로따로 나눠 하는 게 맞는데, 여기선 그냥 검증데이터와 테스트데이터를 같은 느낌으로 넣어주었다고 한다. 
#요기선 early_stopping_rounds = early_stoppings 안해줬넹~

'''
evals : 검증데이터. dtrain, dtest 2개를 넣어주었다. 
사실 이렇게 하면 안됨. train할 때 학습 데이터로 dtrain을 넣어주었는데, 검증을 똑같이 dtrain으로 해줘버리면 안 됨. 
지금 출력된 결과도 train-logloss는 급격히 줄어들어 좋은 결과가 나오지만, eval-logloss는 그것보다는 좋지 않은 결과가 나옴. 
두 결과 중 dtest(eval-logloss)만을 봐야 함. 
'''


# ** predict()를 통해 예측 확률값을 반환하고 예측 값으로 변환 **

# In[8]:


pred_probs = xgb_model.predict(dtest)
print('predict( ) 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
print(np.round(pred_probs[:10],3))

# 예측 확률이 0.5 보다 크면 1 , 그렇지 않으면 0 으로 예측값 결정하여 List 객체인 preds에 저장 
preds = [ 1 if x > 0.5 else 0 for x in pred_probs ]
print('예측값 10개만 표시:',preds[:10])


# ** get_clf_eval( )을 통해 예측 평가 **

# In[9]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

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
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))


# In[10]:


get_clf_eval(y_test , preds, pred_probs)


# ** Feature Importance 시각화 **

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax) #plot_importance : 얘도 당연히 sklearn이 아니라 xgboost에서 가져옴. 트리 만들 때 가장 많이 기여한 순서
                #모델정보, 그래프정보


# ### 사이킷런 Wrapper XGBoost 개요 및 적용 
# 
# ** 사이킷런 래퍼 클래스 임포트, 학습 및 예측 **

# In[11]:


# 사이킷런 래퍼 XGBoost 클래스인 XGBClassifier 임포트
from xgboost import XGBClassifier

evals = [(X_test, y_test)]

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3) #learning_rate : step
xgb_wrapper.fit(X_train , y_train,  early_stopping_rounds=400,eval_set=evals, eval_metric="logloss",  verbose=True)
                #X_train , y_train은 당연히 pandas DataFrame이군! 여기선 다 쓸수 있다~
                #근데 어차피 n_estimators=400인데 여기서 early_stopping_rounds=400이라 하면 아무 의미 없다. 
                #eval_set : 검증데이터, verbose : True-트래킹메시지(loss 출력메시지) 띄워줌, False-출력아무것도없이 실행만됨

w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]


# In[12]:


get_clf_eval(y_test , w_preds, w_pred_proba)


# ** early stopping을 100으로 설정하고 재 학습/예측/평가 **

# In[13]:


from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)

evals = [(X_test, y_test)]
xgb_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                eval_set=evals, verbose=True)

ws100_preds = xgb_wrapper.predict(X_test)
ws100_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]


# In[14]:


get_clf_eval(y_test , ws100_preds, ws100_pred_proba)


# ** early stopping을 10으로 설정하고 재 학습/예측/평가 **

# In[15]:


# early_stopping_rounds를 10으로 설정하고 재 학습. 
xgb_wrapper.fit(X_train, y_train, early_stopping_rounds=10, 
                eval_metric="logloss", eval_set=evals,verbose=True)

ws10_preds = xgb_wrapper.predict(X_test)
ws10_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test , ws10_preds, ws10_pred_proba)


# In[16]:


from xgboost import plot_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(10, 12))
# 사이킷런 래퍼 클래스를 입력해도 무방. 
plot_importance(xgb_wrapper, ax=ax)


# In[ ]:




