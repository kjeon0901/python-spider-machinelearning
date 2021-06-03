import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew # 비대칭도(skewness)에 대한 값 뱉어줌. 

sample_size = 500

X = np.random.normal(0, 5, sample_size) # 평균 0 표준편차 5인 데이터 500개 뽑아줌
    # random.randn(m, n) : 평균 0 표준편차 1인 정규분포를 (m,n)만큼 뽑아준다. 
    # random.normal(평균, 표준편차, size) : 구체적으로 평균과 표준편차를 알려줌!
sns.distplot(X)
print(abs(X.min()))
'''12.134544252814834'''

X = X + abs(X.min()) # |최솟값|을 X벡터에 모두 더해줌 => 최솟값이 0이 되도록 shift !
sns.distplot(X)

r = np.random.normal(0, 2, sample_size)
sns.distplot(r)

Y = X * 1.0 + r + abs(r.min()) 
    # y=x 그래프에 r라는 noise 정규분포를 또 줘서 scatter 주변에 막 찍히게 해줌. 
    # 대신 y값이 음수가 나오지 않도록 r의 최솟값도 또 더해줘서 shift시켜줌. 
plt.scatter(X,Y)

df = pd.DataFrame({'X':X, 'Y':Y})
print(df.head())
'''        X          Y
0  21.482822  31.258772
1  19.002501  26.392140
2  14.705231  25.367837
3  16.741191  22.972485
4  10.085656  22.446603
'''

sns.jointplot(x='X', y='Y', data=df, alpha=0.5)
#plt.savefig('../../assets/images/markdown_img/180605_1519_resolve_skewness_scatter_plot.svg')
plt.show()
'''
jointplot() : scatter 그래프에 X축, Y축에 대한 히스토그램까지 함께 보여줌. 
sample_size가 엄청 커지면 도수분포도 그냥 scatter 그래프로만은 판단하기 어렵기 때문에 유용. 
'''

sqr_vs = [0.2, 0.5, 1.0, 2.0, 3.0]

f, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, j in enumerate(sqr_vs):
    axes[i].set_title('X ** {} \nskewness = {:.2f}'.format(j, skew(df['X']**j)))
    sns.distplot(df['X']**j, ax=axes[i], kde=False)
#plt.savefig('../../assets/images/markdown_img/180605_1517_resolve_skewness_compare.svg')
plt.show()
'''
skewness의 절댓값이 커질수록 비대칭성 커짐. 
skewness가 0에 가까울수록 표준정규분포와 가까워짐. 
'''