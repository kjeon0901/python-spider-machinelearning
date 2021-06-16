#!/usr/bin/env python
# coding: utf-8

# ### 데이터 셋 로딩과 데이터 클린징

# In[1]:


import pandas as pd
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

retail_df = pd.read_excel(io='C:/jeon/Online Retail.xlsx')
retail_df.head(3)
'''
고객 세그먼테이션은 '고객의 어떤 요소를 기반으로 군집화할 것인가?' 가 중요. 
여기서는 RFM 기법을 사용해서 고객을 군집화해볼 것이다. 
R RECENCY           : 가장 최근 상품 구입 일에서 오늘까지의 기간
F FREQUENCY         : 상품 구매 횟수
M MONETARY VALUE    : 총 구매 금액
'''

# In[2]:


retail_df.info()
retail_df.describe()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 541909 entries, 0 to 541908
Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype         
---  ------       --------------   -----         
 0   InvoiceNo    541909 non-null  object          주문번호. 'C'로 시작하면 취소 주문. 
 1   StockCode    541909 non-null  object          제품 코드
 2   Description  540455 non-null  object          제품 설명        => Null값 있음!
 3   Quantity     541909 non-null  int64           주문 제품 수량
 4   InvoiceDate  541909 non-null  datetime64[ns]  주문 일자
 5   UnitPrice    541909 non-null  float64         제품 단가
 6   CustomerID   406829 non-null  float64         고객 번호        => Null값 있음!
 7   Country      541909 non-null  object          국가명 (주문 고객의 국적)
dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
memory usage: 33.1+ MB

            Quantity      UnitPrice     CustomerID      => int, float인 column만 나옴. 
count  541909.000000  541909.000000  406829.000000
mean        9.552250       4.611114   15287.690570
std       218.081158      96.759853    1713.600303
min    -80995.000000  -11062.060000   12346.000000      => Quantity(주문 제품 수량), UnitPrice(제품 단가)가 음수가 나온 게 이상하다!! => 제외하자!!
25%         1.000000       1.250000   13953.000000
50%         3.000000       2.080000   15152.000000
75%        10.000000       4.130000   16791.000000
max     80995.000000   38970.000000   18287.000000
'''


# **반품이나 CustomerID가 Null인 데이터는 제외, 영국 이외 국가의 데이터는 제외**

# In[3]:


retail_df = retail_df[retail_df['Quantity'] > 0]
retail_df = retail_df[retail_df['UnitPrice'] > 0]
retail_df = retail_df[retail_df['CustomerID'].notnull()]
print(retail_df.shape) '''(397884, 8)'''
retail_df.isnull().sum() # column별 null값이 몇 개인지
'''
InvoiceNo      0
StockCode      0
Description    0  => 이상한 거 없애주니까 얘도 (딱히 상관없었지만..) null 값 없어졌넹 ㅎㅎ
Quantity       0
InvoiceDate    0
UnitPrice      0
CustomerID     0
Country        0
dtype: int64
'''

# In[4]:


retail_df['Country'].value_counts()[:20]
'''
United Kingdom     354321 多多多 !
Germany              9040
France               8341
EIRE                 7236
Spain                2484
Netherlands          2359
Belgium              2031
Switzerland          1841
Portugal             1462
Australia            1182
Norway               1071
Italy                 758
Channel Islands       748
Finland               685
Cyprus                614
Sweden                451
Austria               398
Denmark               380
Poland                330
Japan                 321
Name: Country, dtype: int64
'''


# In[5]:


retail_df = retail_df[retail_df['Country']=='United Kingdom'] # 영국 고객이 엄청 많았으니까, 영국 데이터만 갖고 봄. 
print(retail_df.shape)


# ### RFM 기반 데이터 가공

# **구매금액 생성**

# In[6]:


retail_df['sale_amount'] = retail_df['Quantity'] * retail_df['UnitPrice'] # 총 구매 금액 (RFM 中 M)
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)


# In[7]:


print(retail_df['CustomerID'].value_counts().head(5)) # 가장 많은 구매 행위를 한 사람 순으로
'''
17841    7847
14096    5111
12748    4595
14606    2700
15311    2379
Name: CustomerID, dtype: int64
'''
print(retail_df.groupby('CustomerID')['sale_amount'].sum().sort_values(ascending=False)[:5])
    # 동일한 CustomerID 끼리 묶어서, 한 CustomerID로 구매한 모든 sale_amount(총 구매 금액)끼리 다 더한 뒤, 내림차순으로 정렬. 
'''
CustomerID
18102    259657.30
17450    194550.79
16446    168472.50
17511     91062.38
16029     81024.84
Name: sale_amount, dtype: float64
'''

# In[8]:

retail_df.groupby(['InvoiceNo','StockCode'])['InvoiceNo'].count() 
    # count() : null이 아닌 것의 개수. ['InvoiceNo'].count()에서 'InvoiceNo'를 넣든, 'Quantity'를 넣든... 뭘 넣든 상관x
    # quentity가 여러개인 게 아니라, 재주문도 아니라, 장바구니에 굳이 여러 번 담아서 산 횟수는 2 이상으로 세어줌. 
'''
InvoiceNo  StockCode    InvoiceNo
536365     21730        1
           22752        1
           71053        1
           84029E       1
           84029G       1
                       ..
581585     84946        1
581586     20685        1
           21217        1
           22061        1
           23275        1
Name: InvoiceNo, Length: 344435, dtype: int64
'''
retail_df.groupby(['InvoiceNo','StockCode'])['InvoiceNo'].count().mean() # 그럴 일은 거의 없기 때문에 평균이 1에 가까움. 
'''1.028702077315023'''

# **고객 기준으로 RFM - Recency, Frequency, Monetary가공**

# In[9]:


# DataFrame의 groupby() 의 multiple 연산을 위해 agg() 이용
# Recency는 InvoiceDate 컬럼의 max() 에서 데이터 가공
# Frequency는 InvoiceNo 컬럼의 count() , Monetary value는 sale_amount 컬럼의 sum()
aggregations = { # R, F, M
    'InvoiceDate': 'max', # 해당 CustomerID 로 주문한 가장 최근 날짜
    'InvoiceNo': 'count', # 해당 CustomerID 로 몇 번의 주문이 있었는가
    'sale_amount':'sum' # 해당 CustomerID 로 지금까지 결제한 총 구매 금액
}
cust_df = retail_df.groupby('CustomerID').agg(aggregations) # .agg() : DataFrame에서 여러 column에 각각의 기능을 수행해줌. 
                                                            # CustomerID로 묶어서 각 CustomerID별로 aggregations기능 수행한 결과 리턴. 
# groupby된 결과 컬럼값을 Recency, Frequency, Monetary로 변경
cust_df = cust_df.rename(columns = {'InvoiceDate':'Recency',
                                    'InvoiceNo':'Frequency',
                                    'sale_amount':'Monetary'
                                   }
                        )
cust_df = cust_df.reset_index()
cust_df.head(3)


# **Recency를 날짜에서 정수형으로 가공**

# In[10]:


cust_df['Recency'].max()
'''Timestamp('2011-12-09 12:49:00')'''

# In[11]:


import datetime as dt

cust_df['Recency'] = dt.datetime(2011,12,10) - cust_df['Recency'] # 가장 최근 : 가장 값이 작다. 
    # dt.datetime(2011,12,10) - Timestamp('2011-12-09 12:49:00') =>   0 days 12:49:00 나옴. 그치만 하루 더 더해줘야 함. 
    # datatime 끼리의 연산도 가능하다! ★ dt.datetime의 강력함 ★
cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days+1) # 최종적으로, 오늘로부터 며칠 전인지. 
print('cust_df 로우와 컬럼 건수는 ',cust_df.shape)
cust_df.head(3)


# ### RFM 기반 고객 세그먼테이션

# **Recency, Frequency, Monetary 값의 분포도 확인**

# In[12]:


fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
ax1.set_title('Recency Histogram')
ax1.hist(cust_df['Recency'])

ax2.set_title('Frequency Histogram')
ax2.hist(cust_df['Frequency'])

ax3.set_title('Monetary Histogram')
ax3.hist(cust_df['Monetary'])


# In[13]:


cust_df[['Recency','Frequency','Monetary']].describe()


# **K-Means로 군집화 후에 실루엣 계수 평가**

# In[14]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

X_features = cust_df[['Recency','Frequency','Monetary']].values
X_features_scaled = StandardScaler().fit_transform(X_features)

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X_features_scaled)
cust_df['cluster_label'] = labels

print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(X_features_scaled,labels)))


# **K-Means 군집화 후에 실루엣 계수 및 군집을 시각화**

# In[15]:


### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성  
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n'                           'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,                                 facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")


# In[16]:


### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 클러스터링 결과를 시각화 
def visualize_kmeans_plot_multi(cluster_lists, X_features):
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 
    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 입력 데이터의 FEATURE가 여러개일 경우 2차원 데이터 시각화가 어려우므로 PCA 변환하여 2차원 시각화
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_features)
    dataframe = pd.DataFrame(pca_transformed, columns=['PCA1','PCA2'])
    
     # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 KMeans 클러스터링 수행하고 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링으로 클러스터링 결과를 dataframe에 저장. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(pca_transformed)
        dataframe['cluster']=cluster_labels
        
        unique_labels = np.unique(clusterer.labels_)
        markers=['o', 's', '^', 'x', '*']
       
        # 클러스터링 결과값 별로 scatter plot 으로 시각화
        for label in unique_labels:
            label_df = dataframe[dataframe['cluster']==label]
            if label == -1:
                cluster_legend = 'Noise'
            else :
                cluster_legend = 'Cluster '+str(label)           
            axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,                        edgecolor='k', marker=markers[label], label=cluster_legend)

        axs[ind].set_title('Number of Cluster : '+ str(n_cluster))    
        axs[ind].legend(loc='upper right')
    
    plt.show()


# In[17]:


visualize_silhouette([2,3,4,5],X_features_scaled)
visualize_kmeans_plot_multi([2,3,4,5],X_features_scaled)


# **로그 변환 후 재 시각화**

# In[18]:


### Log 변환을 통해 데이터 변환
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Recency, Frequecny, Monetary 컬럼에 np.log1p() 로 Log Transformation
cust_df['Recency_log'] = np.log1p(cust_df['Recency'])
cust_df['Frequency_log'] = np.log1p(cust_df['Frequency'])
cust_df['Monetary_log'] = np.log1p(cust_df['Monetary'])

# Log Transformation 데이터에 StandardScaler 적용
X_features = cust_df[['Recency_log','Frequency_log','Monetary_log']].values
X_features_scaled = StandardScaler().fit_transform(X_features)

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X_features_scaled)
cust_df['cluster_label'] = labels

print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(X_features_scaled,labels)))


# In[19]:


visualize_silhouette([2,3,4,5],X_features_scaled)
visualize_kmeans_plot_multi([2,3,4,5],X_features_scaled)


# In[ ]:




