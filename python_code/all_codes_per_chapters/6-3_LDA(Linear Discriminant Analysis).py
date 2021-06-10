#!/usr/bin/env python
# coding: utf-8

# ### LDA 개요 
# ### 붓꽃 데이터 셋에 LDA 적용하기 

# In[2]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
iris_scaled = StandardScaler().fit_transform(iris.data) # label빼고 피처값만 모두 정규화


# In[3]:

'''
무조건 갖다 쓰는 건 bad. 
항상 뭘 하든 내부에 어떤 일이 있는지, 어떻게 해야 하는지, 내 뜻대로 안 되면 뭐가 문제인지 
개인 시간 투자해서 10시간 걸려도 파고들어야 한다. 
그래야 감이 생김. => 1년, 2년, 연차 쌓일 수록 속도 확연히 빨라짐. 
'''
lda = LinearDiscriminantAnalysis(n_components=2)
# fit()호출 시 target값 입력 
lda.fit(iris_scaled, iris.target)   # 보통은 fit 시킬 때 feature, target 같이 넣어줘야 함. 
                                    # PCA는 각각의 sample들이 어떤 target값을 가지는지 알 필요 x. feature만 넣어도 됐음.
                                    # LDA는 알고리즘 상 target값이 필요함. label데이터 군집 끼리의 분산, 각 label군집 안에서의 분산이 필요. 그래서 target값도 넣어줌. 
iris_lda = lda.transform(iris_scaled)
print(iris_lda.shape)


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

lda_columns=['lda_component_1','lda_component_2']
irisDF_lda = pd.DataFrame(iris_lda,columns=lda_columns)
irisDF_lda['target']=iris.target # irisDF_lda size:(150, 3)

#setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=['^', 's', 'o']

#setosa의 target 값은 0, versicolor는 1, virginica는 2. 각 target 별로 다른 shape으로 scatter plot
for i, marker in enumerate(markers):
    x_axis_data = irisDF_lda[irisDF_lda['target']==i]['lda_component_1']
    y_axis_data = irisDF_lda[irisDF_lda['target']==i]['lda_component_2']

    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend(loc='upper right')
plt.xlabel('lda_component_1')
plt.ylabel('lda_component_2')
plt.show()


# In[ ]:

# PCA 변환한 데이터로 그린 scatter 와 비교. 아래 코드는 PCA. 위에서 그린 LDA가 조금 더 명확하게 분류됨. 

from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 사이킷런 내장 데이터 셋 API 호출
iris = load_iris()

# 넘파이 데이터 셋을 Pandas DataFrame으로 변환
columns = ['sepal_length','sepal_width','petal_length','petal_width']
irisDF = pd.DataFrame(iris.data , columns=columns)
irisDF['target']=iris.target
irisDF.head(3)
from sklearn.preprocessing import StandardScaler

iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:, :-1])
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

#fit( )과 transform( ) 을 호출하여 PCA 변환 데이터 반환
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print(iris_pca.shape)
pca_columns=['pca_component_1','pca_component_2']
irisDF_pca = pd.DataFrame(iris_pca,columns=pca_columns)
irisDF_pca['target']=iris.target
#setosa를 세모, versicolor를 네모, virginica를 동그라미로 표시
markers=['^', 's', 'o']

#pca_component_1 을 x축, pc_component_2를 y축으로 scatter plot 수행. 
for i, marker in enumerate(markers):
    x_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_1']
    y_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()


# In[ ]:




