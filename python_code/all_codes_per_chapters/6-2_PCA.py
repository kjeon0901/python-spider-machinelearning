#!/usr/bin/env python
# coding: utf-8

# ### PCA 개요 

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 사이킷런 내장 데이터 셋 API 호출
iris = load_iris()

# 넘파이 데이터 셋을 Pandas DataFrame으로 변환
columns = ['sepal_length','sepal_width','petal_length','petal_width'] # 4개의 feature
irisDF = pd.DataFrame(iris.data , columns=columns)
irisDF['target']=iris.target # 3개의 label
irisDF.head(3)


# ** sepal_length, sepal_width 두개의 속성으로 데이터 산포 시각화 **

# In[2]:


#setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=['^', 's', 'o']

#setosa의 target 값은 0, versicolor는 1, virginica는 2. 각 target 별로 다른 shape으로 scatter plot 
for i, marker in enumerate(markers):
    x_axis_data = irisDF[irisDF['target']==i]['sepal_length'] # feature 2개만 뽑아서 축 2개로 scatter해봄
    y_axis_data = irisDF[irisDF['target']==i]['sepal_width']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i]) # 3개의 label이 scatter로 그려지는 마커가 서로 다르다. 
    '''
    setosa는 scatter 그래프에서 구분이 명확. 
    versicolor, virginica는 섞여 있는 데이터가 많아서 나중에 모델이 구분 못함. 
    '''

plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()


# ** 평균이 0, 분산이 1인 정규 분포로 원본 데이터를 변환 **

# In[3]:


from sklearn.preprocessing import StandardScaler

iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:, :-1]) # StandardScaler로 정규분포로 표준화해서 label값 제외한 피처데이터만 리턴


# In[4]:


iris_scaled.shape


# ** PCA 변환 수행 **

# In[5]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2) # 2개의 축 만들어서 2차원으로 차원 축소하겠다. pca를 수행하는 객체. 

#fit( )과 transform( ) 을 호출하여 PCA 변환 데이터 반환
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled) # pca가 iris_scaled를 가장 잘 나타내는 축 2개에 데이터 투영해서 리턴
    # 순서1. iris_scaled의 공분산 행렬 만듦
    # 순서2. 가장 높은 고유값 2개 구함
    # 순서3. 그 고유값에 해당하는 2개의 고유벡터(주성분벡터) 구함
    # 순서4. 2개의 주성분벡터에 데이터 투영
print(iris_pca.shape)


# In[6]:


# PCA 변환된 데이터의 컬럼명을 각각 pca_component_1, pca_component_2로 명명
pca_columns=['pca_component_1','pca_component_2']
irisDF_pca = pd.DataFrame(iris_pca,columns=pca_columns)
irisDF_pca['target']=iris.target
irisDF_pca.head(3)


# ** PCA로 차원 축소된 피처들로 데이터 산포도 시각화 **

# In[7]:


#setosa를 세모, versicolor를 네모, virginica를 동그라미로 표시
markers=['^', 's', 'o']

#pca_component_1 을 x축, pc_component_2를 y축으로 scatter plot 수행. 
for i, marker in enumerate(markers):
    x_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_1'] # 이제는 pca 끝난 2개의 축으로 scatter. 
    y_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])
    '''
    setosa는 scatter 그래프에서 구분이 여전히 명확. 
    versicolor, virginica는 원본 데이터보다 훨씬 구분 명확해짐. 
     => 아까보다 모델링 정확도 높일 수 있음. 
     
    즉, 차원 축소한답시고 column 2개만 남기고 다 제거해서 보는 것보다, 
    이렇게 PCA로 제대로 2개의 축 만들어서 모델링하는 게 훨씬 좋다!!!!
    '''

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()


# ** 각 PCA Component별 변동성 비율 **

# In[8]:


print(pca.explained_variance_ratio_)
'''
[0.72962445 0.22850762]
둘을 더하면 0.95~~ => pca로 4개 → 2개의 축으로 차원 축소했지만, irisDF의 전체 데이터를 95%정도는 설명할 수 있다. 
'''


# ** 원본 데이터와 PCA 변환된 데이터 기반에서 예측 성능 비교 **

# In[9]:


from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트 분류로 퍼포먼스 비교
from sklearn.model_selection import cross_val_score # 3번 교차검증
import numpy as np

rcf = RandomForestClassifier(random_state=156)
scores = cross_val_score(rcf, iris.data, iris.target,scoring='accuracy',cv=3)
print(scores)
print(np.mean(scores))
'''
=== 4개의 축 ===
[0.98 0.94 0.96]    //개별정확도
0.96                //평균정확도
'''

# In[10]:


pca_X = irisDF_pca[['pca_component_1', 'pca_component_2']]
scores_pca = cross_val_score(rcf, pca_X, iris.target, scoring='accuracy', cv=3 )
print(scores_pca)
print(np.mean(scores_pca))
'''
=== 2개의 축 ===
[0.88 0.88 0.88]    //개별정확도
0.88                //평균정확도

지금은 정확도 10% 낮아짐. 원래 피처가 4개밖에 없어서 여기선 딱히 별로지만..!
1. 나중에 피처 개수 엄청 늘어나거나
2. 각각의 피처가 상관계수가 높은 피처가 많거나 하면
아주 좋다~~!!!

또한, 시간 압축은 당연히 되지만 '회귀' 관련 문제에서는 '다중 공선성' 기준에서 압축을 통해 퍼포먼스가 좋아질 수 있음!
'''

# ### 신용카드 데이터 세트 PCA 변환      // 이제 PCA변환이 유용할 것으로 예상되는 데이터에 해보자!
# 
# ** 데이터 로드 및 컬럼명 변환 **

# In[11]:


import pandas as pd

df = pd.read_excel('C:/jeon/pca_credit_card.xls', sheet_name='Data', header=1)
print(df.shape) '''(30000, 25) >> 이번엔 컬럼이 25개나!'''
df.head(3)


# In[12]:


df.rename(columns={'PAY_0':'PAY_1','default payment next month':'default'}, inplace=True)
    # PAY_0 다음이 PAY_2니까, 타겟 default payment next month 이름이 너무 기니까 컬럼이름 바꿔줌
y_target = df['default'] # 타겟 'default' 다음달 연체 여부 => 0-연체no, 1-연체yes
# ID, default 컬럼 Drop
X_features = df.drop(['ID','default'], axis=1)


# In[13]:


y_target.value_counts()


# In[14]:


X_features.info() # null값 없음


# ** 피처간 상관도 시각화 **

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corr = X_features.corr() # 상관계수
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True, fmt='.1g')
'''
heapmap 그래프 보면
BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6
이 6개 피처들은 상관계수가 엄청 높은 게 보임. 

=> 얘네를 PCA변환으로 차원축소 하면 되겠다 !!
'''


# **상관도가 높은 피처들의 PCA 변환 후 변동성 확인**

# In[16]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#BILL_AMT1 ~ BILL_AMT6까지 6개의 속성명 생성
cols_bill = ['BILL_AMT'+str(i) for i in range(1, 7)] # list comprehension
print('대상 속성명:', cols_bill)

# 2개의 PCA 속성을 가진 PCA 객체 생성하고, explained_variance_ratio_ 계산을 위해 fit( ) 호출
scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill]) # ID, label 데이터 제외한 피처데이터에서 cols_bill 6개의 컬럼만 정규화
pca = PCA(n_components=2) # 고유값 가장 높은 축 2개로 차원축소하겠다. 
pca.fit(df_cols_scaled) # 6줄의 데이터를 축 2개로 축소시켰다. 

print('PCA Component별 변동성:', pca.explained_variance_ratio_)
'''
PCA Component별 변동성: [0.90555253 0.0509867 ]  => 각 축마다 전체 데이터를 얼만큼 설명하는지. 
총 2개의 축으로 df_cols_scaled의 전체 데이터 6줄을 95%만큼 설명한다.  => 6줄 상관계수가 정말 높았구나~
'''

# ** 원본 데이터 세트와 6개 컴포넌트로 PCA 변환된 데이터 세트로 분류 예측 성능 비교 **

# In[17]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rcf = RandomForestClassifier(n_estimators=300, random_state=156)
scores = cross_val_score(rcf, X_features, y_target, scoring='accuracy', cv=3 ) # PCA변환으로 차원 축소 하기 전
                            # X_features : (30000, 23)

print('CV=3 인 경우의 개별 Fold세트별 정확도:',scores)
print('평균 정확도:{0:.4f}'.format(np.mean(scores)))
'''
=== 25개의 축 ===
[0.8083 0.8196 0.8232]  //개별정확도
0.8170                  //평균정확도
'''

# In[18]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 원본 데이터셋에 먼저 StandardScaler적용
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X_features) # 일단 전체 X_features(ID, default(label) 제거된) 데이터를 모두 정규화 

# 6개의 Component를 가진 PCA 변환을 수행하고 cross_val_score( )로 분류 예측 수행. 
pca = PCA(n_components=6) # 전체 25개의 축을 6개로 축약시키겠다~!
df_pca = pca.fit_transform(df_scaled)
scores_pca = cross_val_score(rcf, df_pca, y_target, scoring='accuracy', cv=3) # PCA 이후
                                # df_pca : (30000, 6)


print('CV=3 인 경우의 PCA 변환된 개별 Fold세트별 정확도:',scores_pca)
print('PCA 변환 데이터 셋 평균 정확도:{0:.4f}'.format(np.mean(scores_pca)))
'''
=== 6개의 축 ===
[0.792  0.7961 0.8053]  //개별정확도
0.7978                  //평균정확도

평균 정확도는 2% 줄었지만, 축이 25 → 6으로 19개나 줄어들었다. 
모든 25개의 축이 전부 다 독립적이었다면 퍼포먼스가 나빠졌겠지만, 상관관계가 높은 축들이 있었기에 크게 나빠지진 않은 것임!!
'''

# In[ ]:




