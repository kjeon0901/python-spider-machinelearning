#!/usr/bin/env python
# coding: utf-8

# ### 붓꽃(Iris) 데이터 셋을 이용한 클러스터 평가

# In[ ]:


from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
# 실루엣 분석 metric 값을 구하기 위한 API 추가
from sklearn.metrics import silhouette_samples, silhouette_score # 2가지 함수를 import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()
feature_names = ['sepal_length','sepal_width','petal_length','petal_width']
irisDF = pd.DataFrame(data=iris.data, columns=feature_names)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300,random_state=0).fit(irisDF) # 3개로 군집화

irisDF['cluster'] = kmeans.labels_

irisDF.head(3)


# In[ ]:


# iris 의 모든 개별 데이터에 실루엣 계수값을 구함. 
score_samples = silhouette_samples(iris.data, irisDF['cluster']) # 전체 데이터 150개의 모든 s(i) 구함
                                                                 # s(i)구하려면 피처데이터iris.data, 데이터i가 어느 cluster에 속해 있는지irisDF['cluster'] 필요. 
print('silhouette_samples( ) return 값의 shape' , score_samples.shape)

# irisDF에 실루엣 계수 컬럼 추가
irisDF['silhouette_coeff'] = score_samples


# In[ ]:


irisDF.head(20)


# In[ ]:


# 모든 데이터의 평균 실루엣 계수값을 구함. 
average_score = silhouette_score(iris.data, irisDF['cluster'])
print('붓꽃 데이터셋 Silhouette Analysis Score:{0:.3f}'.format(average_score))
'''
붓꽃 데이터셋 Silhouette Analysis Score:0.553     → 전체 실루엣 계수의 평균값
'''

# In[ ]:


irisDF['silhouette_coeff'].hist() # 실루엣계수 값 분포 히스토그램


# In[ ]:


irisDF.groupby('cluster')['silhouette_coeff'].mean() # 'cluster'의 유니크한 값별로 (군집별로) 묶어서 각 실루엣 계수의 평균값
'''
cluster
0    0.417320
1    0.798140
2    0.451105
Name: silhouette_coeff, dtype: float64
'''

# ### 클러스터별 평균 실루엣 계수의 시각화를 통한 클러스터 개수 최적화 방법

# In[14]:


### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
# 그래프 이쁘게 그리려고 함수 좀 길게 써진 것임. 
def visualize_silhouette(cluster_lists, X_features): 
                        # [2, 3, 4, 5], X
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols) # (1, 4) 크기로 subplots 그리겠다. 
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        # ind: 0,1,2,3    n_cluster: 2,3,4,5
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0) # k-평균 군집화할 객체
        cluster_labels = clusterer.fit_predict(X_features) # 객체 clusterer로 X_features 군집화 해줌. 각 데이터 별 어떤 군집에 해당되는지 레이블값 담김.
        
        sil_avg = silhouette_score(X_features, cluster_labels) # 전체 실루엣 계수의 평균
        sil_values = silhouette_samples(X_features, cluster_labels) # 전체 실루엣 계수
        
        # plot 설정값들
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n'
                           'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster): # 2, 3, 4, 5바퀴씩 돌 것 → i = 0,1  ,  0,1,2  ,  0,1,2,3  ,  0,1,2,3,4
            ith_cluster_sil_values = sil_values[cluster_labels==i] # 군집이 i인 애들의 실루엣 계수만 모아서
            ith_cluster_sil_values.sort() # 실루엣 계수 정렬
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,
                                   facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
    '''
    전체 실루엣 계수의 평균값 sil_avg 은 n_cluster = 2 일 때 가장 높다. 
    
    BUT !!!
    전체 실루엣 계수의 평균값 sil_avg 과 개별 군집의 실루엣 계수의 평균값 의 차이가 작아야 한다. 
    n_cluster = 2 그래프를 보면, 
    1번 군집은 대체로 sil_avg 보다 커서 좋지만, 그에 따라 반대로 2번 군집은 일부를 제외하고 대체로 sil_avg 보다 낮음. 
    
    OTHERWISE !!!
    n_cluster = 4 그래프를 보면, 
    개별 군집의 평균 실루엣 계수 값이 비교적 균일하므로, n_cluster = 2 일 때보다 sil_avg 가 작지만 조금 더 이상적이다. 
    
    전체 실루엣 계수의 평균값보다 더 중요한 건, 꽤 높은 값의 s(i)들이 고르게 분포되어야 한다는 것이다.
    '''


# In[15]:


# make_blobs 을 통해 clustering 을 위한 4개의 클러스터 중심의 500개 2차원 데이터 셋 생성  
from sklearn.datasets import make_blobs # 테스트 위해서 임의로 샘플 만들기 위한 함수 (군집화된 데이터 샘플 만드는 데 특화됨. )
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, # 총 4개의 군집으로 이루어진 (500, 2) 데이터 샘플을 만듦. 
                  center_box=(-10.0, 10.0), shuffle=True, random_state=1)  

# cluster 개수를 2개, 3개, 4개, 5개 일때의 클러스터별 실루엣 계수 평균값을 시각화 
visualize_silhouette([ 2, 3, 4, 5], X) # → 이 데이터를 몇 개의 cluster로 묶는 게 나은지 살펴보기 위함. 


# In[16]:


from sklearn.datasets import load_iris

iris=load_iris()
visualize_silhouette([ 2, 3, 4,5 ], iris.data)
'''
붓꽃데이터에서도 실루엣계수 각각 살펴봤는데, 얘는 2개로 군집화할 때가 제일 좋을 듯 하다!
'''


# In[13]:


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    '''
    Automatically created module for IPython interactive environment
    For n_clusters = 2 The average silhouette_score is : 0.7049787496083262
    For n_clusters = 3 The average silhouette_score is : 0.5882004012129721
    For n_clusters = 4 The average silhouette_score is : 0.6505186632729437
    For n_clusters = 5 The average silhouette_score is : 0.56376469026194
    For n_clusters = 6 The average silhouette_score is : 0.4504666294372765
    '''

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()
'''
실제로 군집화된 데이터를 scatter로 살펴보니, 아까 생각한 것처럼 n_cluster = 4 로 두는 게 맞는 것 같다.  
'''


# In[ ]:




