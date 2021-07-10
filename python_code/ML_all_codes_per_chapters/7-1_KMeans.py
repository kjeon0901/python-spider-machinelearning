#!/usr/bin/env python
# coding: utf-8

# ### K-Means를 이용한 붓꽃(Iris) 데이터 셋 Clustering

# In[11]:


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()

# 보다 편리한 데이터 Handling을 위해 DataFrame으로 변환
irisDF = pd.DataFrame(data=iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
irisDF.head(3)


# **KMeans 객체를 생성하고 군집화 수행**

# In[12]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300,random_state=0) # 3개의 군집으로 묶겠다. 
kmeans.fit(irisDF)


# **labels_ 속성을 통해 각 데이터 포인트별로 할당된 군집 중심점(Centroid)확인하고 irisDF에 'cluster' 컬럼으로 추가**

# In[15]:


print(kmeans.labels_) # 3개의 군집에 레이블 0, 1, 2 부여
print(kmeans.predict(irisDF)) # 위랑 똑같은 code


# In[16]:


irisDF['cluster']=kmeans.labels_


# In[17]:


irisDF['target'] = iris.target # 실제 결정값 => 내가 군집화로 묶은 'cluster'데이터와 정해진 레이블값이 다름. ex_0, 1, 2 vs 1, 0, 2
iris_result = irisDF.groupby(['target','cluster'])['sepal_length'].count()
    # groupby(by='컬럼명') : 엑셀에서 '정렬'같은 느낌. '컬럼명'의 유니크한 값을 기준으로 묶어서 dataframe 만듦. dataframe groupby는 agg()를 써야 처리하기 쉽다.
    # 우선 'target'에 들어 있는 유니크한 레이블값 0, 1, 2로 묶고, 각 묶음 안에서 'cluster'의 유니크한 레이블값 0, 1, 2로 또 묶음. 
    # ['sepal_length'].count()에서 sepal_length는 별 의미 없음. 그냥 몇 개씩 묶였는지 데이터 개수를 확인하고 싶을 뿐. 
print(iris_result)
'''
target  cluster
0       1          50  → target이 0인 setosa는 군집화가 완벽하다. 
1       0          48  → target이 1인 versicolor도 군집화가 꽤 잘 되었다. 
        2           2
2       0          14  → target이 2인 virginica는 망했다...^^ 
        2          36
Name: sepal_length, dtype: int64
'''

# In[19]:


iris.target_names


# **2차원 평면에 데이터 포인트별로 군집화된 결과를 나타내기 위해 2차원 PCA값으로 각 데이터 차원축소**

# In[20]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)

irisDF['pca_x'] = pca_transformed[:,0] # 차원 축소한 2개 피처 중 첫번째 피처
irisDF['pca_y'] = pca_transformed[:,1] # 차원 축소한 2개 피처 중 두번째 피처
irisDF.head(3)


# In[22]:


plt.scatter(x=irisDF.loc[:, 'pca_x'], y=irisDF.loc[:, 'pca_y'], c=irisDF['cluster']) # c : 색깔 구분
# 축소된 피처(차원) 2개로 그린 2차원
# 원래의 target값이 뭐가 됐든 이 이미지만 봤을 땐, 상당히 합리적으로 군집화가 되었다. 

# In[21]:


# cluster 값이 0, 1, 2 인 경우마다 별도의 Index로 추출
marker0_ind = irisDF[irisDF['cluster']==0].index
marker1_ind = irisDF[irisDF['cluster']==1].index
marker2_ind = irisDF[irisDF['cluster']==2].index

# cluster값 0, 1, 2에 해당하는 Index로 각 cluster 레벨의 pca_x, pca_y 값 추출. o, s, ^ 로 marker 표시
plt.scatter(x=irisDF.loc[marker0_ind,'pca_x'], y=irisDF.loc[marker0_ind,'pca_y'], marker='o') 
plt.scatter(x=irisDF.loc[marker1_ind,'pca_x'], y=irisDF.loc[marker1_ind,'pca_y'], marker='s')
plt.scatter(x=irisDF.loc[marker2_ind,'pca_x'], y=irisDF.loc[marker2_ind,'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusters Visualization by 2 PCA Components')
plt.show()


# ### Clustering(군집화) 알고리즘 테스트를 위한 데이터 생성 

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
print(X.shape, y.shape)
'''(200, 2) (200,)'''

# y target 값의 분포를 확인
unique, counts = np.unique(y, return_counts=True) # y 안에 unique한 데이터는 무엇이 있고, 각각 몇 개 있는지 리턴
print(unique,counts)
'''[0 1 2] [67 67 66]  → 200개가 골고루 3개로 잘 나눠졌구나~ '''

# * n_samples: 생성할 총 데이터의 개수입니다. 디폴트는 100개입니다.
#  
#  
# * n_features: 데이터의 피처 개수입니다. 시각화를 목표로 할 경우 2개로 설정해 보통 첫 번째 피처는 x 좌표, 두 번째 피처 
# 는 y 좌표상에 표현합니다.   
#  
#  
# * centers: int 값, 예를 들어 3으로 설정하면 군집의 개수를 나타냅니다. 그렇지 않고 ndarray 형태로 표현할 경우 개별 군 
# 집 중심점의 좌표를 의미합니다.   
#  
#  
# * cluster_std: 생성될 군집 데이터의 표준 편차를 의미합니다. 만일 float 값 0.8과 같은 형태로 지정하면 군집 내에서 데이 
# 터가 표준편차 0.8을 가진 값으로 만들어집니다.    
# [0.8, 1,2, 0.6]과 같은 형태로 표현되면 3개의 군집에서 첫 번째 군집 내 
# 데이터의 표준편차는 0.8, 두 번째 군집 내 데이터의 표준 편차는 1.2, 세 번째 군집 내 데이터의 표준편차는 0.6으로 만듭 
# 니다.    
# 군집별로 서로 다른 표준 편차를 가진 데이터 세트를 만들 때 사용합니다   

# In[24]:


import pandas as pd

clusterDF = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y
clusterDF.head(3)


# **make_blob()으로 만들어진 데이터 포인트들을 시각화**

# In[25]:


target_list = np.unique(y) # target_list = [0 1 2] 들어감. 
# 각 target별 scatter plot 의 marker 값들. 
markers=['o', 's', '^', 'P','D','H','x']
# 3개의 cluster 영역으로 구분한 데이터 셋을 생성했으므로 target_list는 [0,1,2]
# target==0, target==1, target==2 로 scatter plot을 marker별로 생성. 
for target in target_list:
    target_cluster = clusterDF[clusterDF['target']==target] # target값 0, 1, 2인 데이터끼리 따로 가져와서 각 target별로
    plt.scatter(x=target_cluster['ftr1'], y=target_cluster['ftr2'], edgecolor='k', marker=markers[target] ) # 'o', 's', '^'라는 marker를 이용해 x축, y축 scatter한다. 
plt.show()


# In[26]:


target_list = np.unique(y) # target_list = [0 1 2] 들어감. 
plt.scatter(x=clusterDF['ftr1'], y=clusterDF['ftr2'], edgecolor='k', c=y ) # 이건 실제 결정값 scatter 찍어봄. 


# **K-Means 클러스터링(군집화)을 수행하고 개별 클러스터의 중심 위치를 시각화**

# In[27]:


# KMeans 객체를 이용하여 X 데이터를 3개의 군집으로 K-Means 클러스터링 수행 
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, random_state=0)
cluster_labels = kmeans.fit_predict(X) # 이 안에서 반복 다 끝나고, 최종 중심점+소속된 데이터 구하고 군집화 끝남. 
clusterDF['kmeans_label']  = cluster_labels # 실제 target값과 값은 다를 수 있지만 어쨌든 군집화는 잘 됨. 
                                            # target 0  kmeans_label 0
                                            #        1               2
                                            #        2               1

#cluster_centers_ 는 개별 클러스터의 중심 위치 좌표 시각화를 위해 추출
centers = kmeans.cluster_centers_ # kmeans 초기화된 3개의 중심점 좌표
unique_labels = np.unique(cluster_labels) # 표시는 0 1 2로 할거다. 
markers=['o', 's', '^', 'P','D','H','x']

# 군집된 label 유형별로 iteration 하면서 marker 별로 scatter plot 수행. 
for label in unique_labels: # 군집(cluster)별로 for문 수행
    label_cluster = clusterDF[clusterDF['kmeans_label']==label]
    center_x_y = centers[label] # 군집 별로 중심점 좌표 담김
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], edgecolor='k', 
                marker=markers[label] ) # 각 데이터를 scatter 찍는다. 
    
    # 군집별 중심 위치 좌표 시각화 
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='white',
                alpha=0.9, edgecolor='k', marker=markers[label]) # 중심점 0, 1, 2 scatter 'o', 's', '^'로 찍는다. 
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k', # s:size. 키우면 scatter 표기 크기 커짐. 
                marker='$%d$' % label) # 중심점마다 scatter 찍은 그 위치에 label값, 즉 0, 1, 2 표기

plt.show()


# In[28]:


kmeans.cluster_centers_
''' 최종 중심점 찍힌 좌표
array([[ 0.990103  ,  4.44666506],
       [-1.70636483,  2.92759224],
       [ 1.95763312,  0.81041752]])
'''


# In[30]:


print(clusterDF.groupby('target')['kmeans_label'].value_counts())
'''
target  kmeans_label
0       0               66
        1                1
1       2               67
2       1               65
        2                1
Name: kmeans_label, dtype: int64
'''


# In[ ]:




