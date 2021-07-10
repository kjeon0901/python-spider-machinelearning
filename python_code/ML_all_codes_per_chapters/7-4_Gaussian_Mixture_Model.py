#!/usr/bin/env python
# coding: utf-8

# ### GMM 을 이용한 붓꽃 데이터 셋 클러스터링

# In[1]:


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()
feature_names = ['sepal_length','sepal_width','petal_length','petal_width']

# 보다 편리한 데이타 Handling을 위해 DataFrame으로 변환
irisDF = pd.DataFrame(data=iris.data, columns=feature_names)
irisDF['target'] = iris.target


# **GaussianMixture를 이용하여 붓꽃 데이터 군집화**

# In[2]:


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=0).fit(iris.data) # 3개의 랜덤 확률분포로 fit
gmm_cluster_labels = gmm.predict(iris.data) # 군집화 결과 label값

# 클러스터링 결과를 irisDF 의 'gmm_cluster' 컬럼명으로 저장
irisDF['gmm_cluster'] = gmm_cluster_labels


# target 값에 따라서 gmm_cluster 값이 어떻게 매핑되었는지 확인. 
iris_result = irisDF.groupby(['target'])['gmm_cluster'].value_counts()
print(iris_result)
'''
target  gmm_cluster
0       0              50
1       2              45
        1               5
2       1              50
Name: gmm_cluster, dtype: int64
'''


# **붓꽃 데이터 K-Means 군집화 결과**

# In[4]:

iris.target_names

# In[3]:

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300,random_state=0).fit(iris.data)
kmeans_cluster_labels = kmeans.predict(iris.data)
irisDF['kmeans_cluster'] = kmeans_cluster_labels
iris_result = irisDF.groupby(['target'])['kmeans_cluster'].value_counts()
print(iris_result)
'''
target  kmeans_cluster
0       1                 50
1       0                 48
        2                  2
2       2                 36
        0                 14
Name: kmeans_cluster, dtype: int64

=> K-Means 알고리즘 특성 상, 중심점에서 원 형태로 군집화할 수밖에 없음. 이 데이터셋에선 나쁜 군집화 방법. 
'''

# **클러스터링 결과를 시각화하는 함수 생성**

# In[6]:


### 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 인자로 받아 클러스터링 결과를 시각화하는 함수  
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
                         # None        clusterDF, 'target',   iscenter=False
    
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = np.unique(dataframe[label_name].values) # 군집화된 유니크한 레이블값 0, 1, 2 전부 담음
    markers=['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels: # 군집화 별로
        label_cluster = dataframe[dataframe[label_name]==label] # 클러스터 결과를 담은 DataFrame에서 label_name이 label인 데이터들만 담음
        if label == -1: # 잘못 나온 label - 일단 이번 데이터셋에는 -1 없음. 
            cluster_legend = 'Noise'
            isNoise=True
        else :
            cluster_legend = 'Cluster '+str(label) # 'Cluster 0', 'Cluster 1', 'Cluster 2' 
        
        # for문 돌면서 각 레이블 별 데이터 scatter 찍어줌
        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70,
                    edgecolor='k', marker=markers[label], label=cluster_legend)
        
        # 레이블 이름도 같이 띄워주고 싶을 때
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',
                        edgecolor='k', marker='$%d$' % label)
    if isNoise:
        legend_loc='upper center'
    else: legend_loc='upper right'
    
    plt.legend(loc=legend_loc)
    plt.show()


# **GMM군집화와 K-Means군집화를 비교하기 위해 타원형으로 늘어선 임의의 데이터 세트를 생성**

# In[7]:


from sklearn.datasets import make_blobs

# make_blobs() 로 300개의 데이터 셋, 3개의 cluster 셋, cluster_std=0.5 을 만듬. 
X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=0)
plt.scatter(X[:, 0], X[:, 1]) # 이 데이터가

# 길게 늘어난 타원형의 데이터 셋을 생성하기 위해 변환함. 
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation) # (2x2).즉 차원축소 때의 개념과 이어 보자면 2개의 축(2개의 column)에다가 데이터 300개를 투영시켜서 
plt.scatter(X_aniso[:, 0], X_aniso[:, 1]) # 이렇게 새로운 데이터 (300, 2) 로 변한다. 
    # 여기서 새로 그려진 x축, y축은 각각 투영시킨 그 벡터. 즉, x = (0.60834549, -0.40887718), y = (-0.63667341, 0.85253229)

# feature 데이터 셋과 make_blobs( ) 의 y 결과 값을 DataFrame으로 저장
clusterDF = pd.DataFrame(data=X_aniso, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y
# 생성된 데이터 셋을 target 별로 다른 marker 로 표시하여 시각화 함. 
visualize_cluster_plot(None, clusterDF, 'target', iscenter=False)


# **K-Means 군집화 수행**

# In[8]:


# 3개의 Cluster 기반 Kmeans 를 X_aniso 데이터 셋에 적용 
kmeans = KMeans(3, random_state=0)
kmeans_label = kmeans.fit_predict(X_aniso)
clusterDF['kmeans_label'] = kmeans_label

visualize_cluster_plot(kmeans, clusterDF, 'kmeans_label',iscenter=True) # K-평균으로 군집화한 데이터 찍어봄. 
'''
K-Means 알고리즘 특성 상, 중심점에서 원 형태로 군집화할 수밖에 없음. 이 데이터셋에선 나쁜 군집화 방법. 
'''


# **Mean Shift 군집화 수행**

# In[9]:


from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

best_bandwidth = estimate_bandwidth(X_aniso)
meanshift = MeanShift(best_bandwidth)
cluster_labels = meanshift.fit_predict(X_aniso)
clusterDF['meanshift_label'] = cluster_labels

visualize_cluster_plot(meanshift, clusterDF, 'meanshift_label',iscenter=True)

    
# **GMM 군집화 수행**

# In[10]:

# 3개의 n_components기반 GMM을 X_aniso 데이터 셋에 적용 
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_label = gmm.fit(X_aniso).predict(X_aniso)
clusterDF['gmm_label'] = gmm_label

# GaussianMixture는 cluster_centers_ 속성이 없으므로 iscenter를 False로 설정. 
visualize_cluster_plot(gmm, clusterDF, 'gmm_label',iscenter=False)
'''
GMM 알고리즘으로 군집화해보니, 원하던 대로 잘 됐다! 
'''

# **GMM과 K-Means 군집화 결과 비교**

# In[11]:


print('### KMeans Clustering ###')
print(clusterDF.groupby('target')['kmeans_label'].value_counts())
print('\n### Mean Shift Clustering ###')
print(clusterDF.groupby('target')['meanshift_label'].value_counts())
print('\n### Gaussian Mixture Clustering ###')
print(clusterDF.groupby('target')['gmm_label'].value_counts())
'''
### KMeans Clustering ###               => BAD :(
target  kmeans_label
0       2                73
        0                27
1       1               100
2       0                86
        2                14
Name: kmeans_label, dtype: int64

### Mean Shift Clustering ###           => BAD :(
target  meanshift_label
0       0                  100
1       1                  100
2       0                  100
Name: meanshift_label, dtype: int64

### Gaussian Mixture Clustering ###     => GOOD ~! :)
target  gmm_label
0       2            100
1       1            100
2       0            100
Name: gmm_label, dtype: int64
'''

# In[ ]:




