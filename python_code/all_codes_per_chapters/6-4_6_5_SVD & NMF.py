#!/usr/bin/env python
# coding: utf-8

# ## 6-4 SVD

# ### SVD 개요

# In[3]:


# numpy의 svd 모듈 import
import numpy as np
from numpy.linalg import svd

# 4X4 Random 행렬 a 생성 - 행렬 개별 row끼리의 의존성 없애려고. 
np.random.seed(121)
a = np.random.randn(4,4) # 정규분포 상에서 4x4 크기에 해당하는 데이터 개수만큼 랜덤값 뽑기
print(np.round(a, 3))
'''
[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.014  0.63   1.71  -1.327]
 [ 0.402 -0.191  1.404 -1.969]]
'''

# **SVD 행렬 분해**

# In[4]:


U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
'''
(4, 4) (4,) (4, 4) → ∑는 1차원으로 리턴되었다. 
어차피 대각선에 있는 값들만 필요하기 때문에 걔네만 이렇게 받아도 ㄱㅊ
'''
print('U matrix:\n',np.round(U, 3))
print('Sigma Value:\n',np.round(Sigma, 3))
print('V transpose matrix:\n',np.round(Vt, 3))
'''
U matrix:
 [[-0.079 -0.318  0.867  0.376]
 [ 0.383  0.787  0.12   0.469]
 [ 0.656  0.022  0.357 -0.664]
 [ 0.645 -0.529 -0.328  0.444]]
Sigma Value:
 [3.423 2.023 0.463 0.079]      → 우리가 나중에 대각행렬 꼴로 바꿔주면 됨. 
V transpose matrix:
 [[ 0.041  0.224  0.786 -0.574]
 [-0.2    0.562  0.37   0.712]
 [-0.778  0.395 -0.333 -0.357]
 [-0.593 -0.692  0.366  0.189]]
'''

# **분해가 잘 되었는지 '검증' => 분해된 행렬들을 이용하여 다시 원행렬로 원복**

# In[5]:


# Sima를 다시 0 을 포함한 대칭행렬로 변환
Sigma_mat = np.diag(Sigma)
print(Sigma_mat) # 원래대로 대각행렬 꼴로 바꿔줌. 
'''
[[3.4229581  0.         0.         0.        ]
 [0.         2.02287339 0.         0.        ]
 [0.         0.         0.46263157 0.        ]
 [0.         0.         0.         0.07935069]]
'''
a_ = np.dot(np.dot(U, Sigma_mat), Vt) # 행렬 3개 차례대로 내적(점곱) 해줌. 
print(np.round(a_, 3))
'''
[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.014  0.63   1.71  -1.327]
 [ 0.402 -0.191  1.404 -1.969]]

=> 음~ 처음의 a행렬과 같군! 진짜로 SVD 기법으로 A = U∑(V^T) 세 개로 쪼개지는구나!
'''


# **데이터 의존도가 높은 원본 데이터 행렬 생성**

# In[6]:


a[2] = a[0] + a[1]
a[3] = a[0]
print(np.round(a,3)) # 이번엔 무작위가 아니라, 개별 row끼리의 의존성 줌. 어떻게 분해될지 아까와 비교해보자!
'''
[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.542  0.899  1.041 -0.073]
 [-0.212 -0.285 -0.574 -0.44 ]]
'''


# In[7]:


# 다시 SVD를 수행하여 Sigma 값 확인 => 특히 ∑의 변화에 집중해보자!
U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
'''
(4, 4) (4,) (4, 4) → 여전히 ∑는 1차원으로 리턴되었다. 
어차피 대각선에 있는 값들만 필요하기 때문에 걔네만 이렇게 받아도 ㄱㅊ
'''
print('Sigma Value:\n',np.round(Sigma,3))
'''
Sigma Value:
 [2.663 0.807 0.    0.   ]       → 이번엔 ∑에 0값이 포함되어 있다!
                                 → row끼리 의존성이 있는 A를 쪼갰을 때 나오는 ∑는 뒤로 갈수록 0에 수렴하는 값을 보인다!
                                 → 그래서 이게 차원 축소와 무슨 상관? => 뒤의 0은 제거해도 됨!!
'''

# In[8]:


# U 행렬의 경우는 Sigma와 내적을 수행하므로 Sigma의 앞 2행에 대응되는 앞 2열만 추출
U_ = U[:, :2]
Sigma_ = np.diag(Sigma[:2])
# V 전치 행렬의 경우는 앞 2행만 추출
Vt_ = Vt[:2]
print(U_.shape, Sigma_.shape, Vt_.shape)
# U, Sigma, Vt의 내적을 수행하며, 다시 원본 행렬 복원
a_ = np.dot(np.dot(U_,Sigma_), Vt_)
print(np.round(a_, 3))


# * Truncated SVD 를 이용한 행렬 분해

# In[10]:


import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

# 원본 행렬을 출력하고, SVD를 적용할 경우 U, Sigma, Vt 의 차원 확인 
np.random.seed(121)
matrix = np.random.random((6, 6))
print('원본 행렬:\n',matrix)
U, Sigma, Vt = svd(matrix, full_matrices=False)
print('\n분해 행렬 차원:',U.shape, Sigma.shape, Vt.shape)
print('\nSigma값 행렬:', Sigma)

# Truncated SVD로 Sigma 행렬의 특이값을 4개로 하여 Truncated SVD 수행. 
num_components = 5
U_tr, Sigma_tr, Vt_tr = svds(matrix, k=num_components)
print('\nTruncated SVD 분해 행렬 차원:',U_tr.shape, Sigma_tr.shape, Vt_tr.shape)
print('\nTruncated SVD Sigma값 행렬:', Sigma_tr)
matrix_tr = np.dot(np.dot(U_tr,np.diag(Sigma_tr)), Vt_tr)  # output of TruncatedSVD

print('\nTruncated SVD로 분해 후 복원 행렬:\n', matrix_tr)


# ### 사이킷런 TruncatedSVD 클래스를 이용한 변환

# In[11]:


from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()
iris_ftrs = iris.data
# 2개의 주요 component로 TruncatedSVD 변환
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_ftrs)
iris_tsvd = tsvd.transform(iris_ftrs)

# Scatter plot 2차원으로 TruncatedSVD 변환 된 데이터 표현. 품종은 색깔로 구분
plt.scatter(x=iris_tsvd[:,0], y= iris_tsvd[:,1], c= iris.target)
plt.xlabel('TruncatedSVD Component 1')
plt.ylabel('TruncatedSVD Component 2')


# In[12]:


from sklearn.preprocessing import StandardScaler

# iris 데이터를 StandardScaler로 변환
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_ftrs)

# 스케일링된 데이터를 기반으로 TruncatedSVD 변환 수행 
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_scaled)
iris_tsvd = tsvd.transform(iris_scaled)

# 스케일링된 데이터를 기반으로 PCA 변환 수행 
pca = PCA(n_components=2)
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

# TruncatedSVD 변환 데이터를 왼쪽에, PCA변환 데이터를 오른쪽에 표현 
fig, (ax1, ax2) = plt.subplots(figsize=(9,4), ncols=2)
ax1.scatter(x=iris_tsvd[:,0], y= iris_tsvd[:,1], c= iris.target)
ax2.scatter(x=iris_pca[:,0], y= iris_pca[:,1], c= iris.target)
ax1.set_title('Truncated SVD Transformed')
ax2.set_title('PCA Transformed')


# ## 6-4 NMF

# ### NMF 

# In[13]:


from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()
iris_ftrs = iris.data
nmf = NMF(n_components=2)

nmf.fit(iris_ftrs)
iris_nmf = nmf.transform(iris_ftrs)

plt.scatter(x=iris_nmf[:,0], y= iris_nmf[:,1], c= iris.target)
plt.xlabel('NMF Component 1')
plt.ylabel('NMF Component 2')


# In[ ]:




