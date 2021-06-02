#!/usr/bin/env python
# coding: utf-8

# ## 로지스틱 회귀

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()


# In[20]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# StandardScaler( )로 평균이 0, 분산 1로 데이터 분포도 변환
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train , X_test, y_train , y_test = train_test_split(data_scaled, cancer.target, test_size=0.3, random_state=0)


# In[21]:


from sklearn.metrics import accuracy_score, roc_auc_score

# 로지스틱 회귀를 이용하여 학습 및 예측 수행. 
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_test)

# accuracy와 roc_auc 측정
print('accuracy: {:0.3f}'.format(accuracy_score(y_test, lr_preds)))
print('roc_auc: {:0.3f}'.format(roc_auc_score(y_test , lr_preds)))


# In[22]:


from sklearn.model_selection import GridSearchCV # 교차검증 수행하며 자동적으로 최적의 파라미터 찾아줌

params={'penalty':['l2', 'l1'], 
        'C':[0.01, 0.1, 1, 1, 5, 10]} 
# 규제 유형을 l2(ridge) or l1(lasso) 로 하겠다
# C = 1/alpha  ==>  alpha = [100, 10, 1, 1, 0.2, 0.1]

grid_clf = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=3 )
grid_clf.fit(data_scaled, cancer.target)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))


# ## 5.8 회귀 트리

# In[ ]:


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# 보스턴 데이터 세트 로드
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

bostonDF['PRICE'] = boston.target
y_target = bostonDF['PRICE'] # 레이블 데이터 세트 - 나중에 train_test_split 할 것
X_data = bostonDF.drop(['PRICE'], axis=1,inplace=False) # 피처 데이터 세트 - 나중에 train_test_split 할 것


rf = RandomForestRegressor(random_state=0, n_estimators=1000) # 랜덤포레스트 트리를 기반으로 회귀 estimator rf 선언
neg_mse_scores = cross_val_score(rf, X_data, y_target, scoring="neg_mean_squared_error", cv = 5) # rf로 k폴드 교차검증
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(' 5 교차 검증의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 교차 검증의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 교차 검증의 평균 RMSE : {0:.3f} '.format(avg_rmse))


# In[ ]:


def get_model_cv_prediction(model, X_data, y_target): # 각 모델, X_data, y_target마다 train_test_split 해서 교차검증 후 퍼포먼스 결과까지 출력하는 함수
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv = 5) # 여기에서 5개의 교차검증마다 X_train, y_train, X_test, y_test 알아서 나눠지고 모델 만들어짐. 
    rmse_scores  = np.sqrt(-1 * neg_mse_scores)
    avg_rmse = np.mean(rmse_scores) # 교차검증이니까 각 fold마다 나온 결과를 평균해줌
    print('##### ',model.__class__.__name__ , ' #####')
    print(' 5 교차 검증의 평균 RMSE : {0:.3f} '.format(avg_rmse))


# ** 사이킷런의 여러 회귀 트리 클래스를 이용하여 회귀 예측 **

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
gb_reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
xgb_reg = XGBRegressor(n_estimators=1000)
lgb_reg = LGBMRegressor(n_estimators=1000)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행 
models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg] # estimator 클래스 객체들 
for model in models:  
    get_model_cv_prediction(model, X_data, y_target) # 아까 위에서 X_data:레이블 데이터 세트, y_target:피처 데이터 세트 둘로 나눔 (train_test_split은 안에서!)
'''
#####  DecisionTreeRegressor  #####
 5 교차 검증의 평균 RMSE : 5.978 
#####  RandomForestRegressor  #####
 5 교차 검증의 평균 RMSE : 4.423 
#####  GradientBoostingRegressor  #####
 5 교차 검증의 평균 RMSE : 4.269 
#####  XGBRegressor  #####
 5 교차 검증의 평균 RMSE : 4.251 
#####  LGBMRegressor  #####
 5 교차 검증의 평균 RMSE : 4.646 
'''



# ** 회귀 트리는 선형 회귀의 회귀 계수 대신, 피처 중요도로 피처의 상대적 중요도를 알 수 있습니다. **

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

rf_reg = RandomForestRegressor(n_estimators=1000)

# 앞 예제에서 만들어진 X_data, y_target 데이터 셋을 적용하여 학습합니다.   
rf_reg.fit(X_data, y_target)

feature_series = pd.Series(data=rf_reg.feature_importances_, index=X_data.columns )
    # feature_importances_ : 트리 만들 때 어떤 feature가 가장 결정적으로 영향을 미쳤는가 (예전에 이거 막대그래프로 만들어봤었음)
feature_series = feature_series.sort_values(ascending=False) # ascending :오름차순. True-오름차순, Falst-내림차순
sns.barplot(x= feature_series, y=feature_series.index) # 이번에도 순서대로 barplot 막대그래프 그림. 
'''
RM: 방개수, LSTAT: 빈곤층..
트리 기반의 회귀를 사용했고, 그 결과로 RM, LSTAT의 영향력이 아주 크다는 걸 알 수 있다. 
생각해보면 그 전 선형 회귀에서도 얘네가 꽤 높은 비중이었었다. 
'''

# ** 오버피팅을 시각화 하기 위해 한개의 피처 RM과 타겟값 PRICE기반으로 회귀 예측 수행 **

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

bostonDF_sample = bostonDF[['RM','PRICE']] # shape : (506, 2). 피처값은 연습이니까 보기 편하게 RM 하나만 가져옴. (PRICE는 target데이터)
bostonDF_sample = bostonDF_sample.sample(n=100,random_state=0) # 랜덤하게 100개만 샘플링해서 리턴
print(bostonDF_sample.shape) # shape : (100, 2)
plt.figure()
plt.scatter(bostonDF_sample.RM , bostonDF_sample.PRICE,c="darkorange") # x축-RM, y축-PRICE
# plt.scatter(bostonDF_sample['RM'] , bostonDF_sample['PRICE'],c="darkorange") 얘와 똑같음. 


# In[ ]:


import numpy as np
from sklearn.linear_model import LinearRegression

# 선형 회귀와 결정 트리 기반의 Regressor 생성. DecisionTreeRegressor의 max_depth는 각각 2, 7
lr_reg = LinearRegression()
rf_reg2 = DecisionTreeRegressor(max_depth=2) # Classifier때와 똑같이 max_depth 정해줘서 과적합 막을 수 있음. 
rf_reg7 = DecisionTreeRegressor(max_depth=7)

# 실제 예측을 적용할 테스트용 데이터 셋을 4.5 ~ 8.5 까지 100개 데이터 셋 생성. 
X_test = np.arange(4.5, 8.5, 0.04).reshape(-1, 1) # X_test. 1차원데이터를 2차원으로 바꿈. 

# 보스턴 주택가격 데이터에서 시각화를 위해 피처는 RM만, 그리고 결정 데이터인 PRICE 추출
X_feature = bostonDF_sample['RM'].values.reshape(-1,1) # X_train
y_target = bostonDF_sample['PRICE'].values.reshape(-1,1) # y_train

# 학습과 예측 수행. 
lr_reg.fit(X_feature, y_target) # RM에 따라 가장 낮은 MSE값을 갖도록 하는 회귀선 만들 것임. 
rf_reg2.fit(X_feature, y_target) # RM에다가 조건식(트리의 노드)을 막 부여해서 ~이만큼 잘라봐, ~이만큼 잘라봐.. ~요기의 평균값, ~요기의 평균값... 
rf_reg7.fit(X_feature, y_target)

pred_lr = lr_reg.predict(X_test)
pred_rf2 = rf_reg2.predict(X_test)
pred_rf7 = rf_reg7.predict(X_test)


# In[ ]:


fig , (ax1, ax2, ax3) = plt.subplots(figsize=(14,4), ncols=3) # 하나의 그림판에 3개의 subplot 그릴 것임. 

# X축값을 4.5 ~ 8.5로 변환하며 입력했을 때, 선형 회귀와 결정 트리 회귀 예측 선 시각화
# 선형 회귀로 학습된 모델 회귀 예측선 
ax1.set_title('Linear Regression')
ax1.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange") # y_test를 scatter찍음. y_test는 동일하므로 셋 다 scatter는 똑같음. 
ax1.plot(X_test, pred_lr,label="linear", linewidth=2 )

# DecisionTreeRegressor의 max_depth를 2로 했을 때 회귀 예측선 
ax2.set_title('Decision Tree Regression: \n max_depth=2')
ax2.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax2.plot(X_test, pred_rf2, label="max_depth:3", linewidth=2 )

# DecisionTreeRegressor의 max_depth를 7로 했을 때 회귀 예측선 
ax3.set_title('Decision Tree Regression: \n max_depth=7')
ax3.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax3.plot(X_test, pred_rf7, label="max_depth:7", linewidth=2)


# In[ ]:




