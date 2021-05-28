#!/usr/bin/env python
# coding: utf-8

# ## 5.3 Gradient Descent

# ** 실제값을 Y=4X+6 시뮬레이션하는 데이터 값 생성 **

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
# y = 4X + 6 식을 근사(w1=4, w0=6). random 값은 Noise를 위해 만듦
X = 2 * np.random.rand(100,1) # 0~1 사이에서 균일분포 기반으로 100개 랜덤값 샘플링. 각각의 요소에 x2 해줌.  
y = 6 + 4*X + np.random.randn(100,1)
'''
random.randint(m, n) : m ~ n-1 사이의 랜덤 숫자 1개 뽑기. m 안써주면 0부터~
random.rand(m, n) : 0 ~ 1 의 '균일분포 표준정규분포' 랜덤 숫자를 (m, n) 크기로 생성.
        균일분포 == 상수함수(연속) : 샘플링한 값의 range가 0 ~ 1이고 0.3 ~ 0,4 사이의 그래프와 x축간의 면적은 "0.3 ~ 0.4 사이의 값이 뽑힐 확률"
                                    떨어진 간극만 같다면 걔네의 면적, 즉 확률도 같다. 
random.randn(m, n) : 평균 0, 표준편차 1의 '가우시안 표준정규분포' 랜덤 숫자를 (m, n) 크기로 생성
        정규분포 : 0.3 ~ 0.4 사이의 값이 뽑힐 확률과 0.4 ~ 0.5 사이의 값이 뽑힐 확률이 다르다. ex_히스토그램 distplot(정규분포, kde=True)
'''
#test = np.random.rand(100000,1)
#test2 = np.random.randn(1000,1)
#sns.distplot(test, kde=False)  #균일분포
#sns.distplot(test2, kde=False) #정규분포

# X, y 데이터 셋 scatter plot으로 시각화
plt.scatter(X, y)


# In[ ]:


X.shape, y.shape


# ** w0과 w1의 값을 최소화 할 수 있도록 업데이트 수행하는 함수 생성.**
# 
# * 예측 배열 y_pred는 np.dot(X, w1.T) + w0 임
# 100개의 데이터 X(1,2,...,100)이 있다면 예측값은 w0 + X(1)*w1 + X(2)*w1 +..+ X(100)*w1이며, 이는 입력 배열 X와 w1 배열의 내적임.
# * 새로운 w1과 w0를 update함
# ![](./image01.png)

# In[ ]:


# w1 과 w0 를 업데이트 할 w1_update, w0_update를 반환. 
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y)
    # 먼저 w1_update, w0_update를 각각 w1, w0의 shape와 동일한 크기를 가진 0 값으로 초기화
    w1_update = np.zeros_like(w1) # np.zeros_like(변수) : 그 변수와 같은 size를 0으로 채워서 초기화하고 리턴.  
    w0_update = np.zeros_like(w0)
    # 예측 배열 계산하고 예측과 실제 값의 차이 계산
    y_pred = np.dot(X, w1.T) + w0 # [x1*w1 + w0, x2*w2 + w0, ... , x100*w100 + w0] → (100, 1) shape
    diff = y-y_pred
         
    # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성 
    w0_factors = np.ones((N,1))

    # w1과 w0을 업데이트할 w1_update와 w0_update 계산  // Δwj = -ηΔJ(w) = -ηJ'(w)
    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff)) # ΔJ(w) = dJ/dwj = J'(w) = ㅡΣ( y(i) - y^(i) ) * xj(i) = ㅡ(Y-Y^)ㆍX  ==  np.dot(X.T, diff)
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))    
    
    return w1_update, w0_update


# In[ ]:


w0 = np.zeros((1,1))
w1 = np.zeros((1,1))
y_pred = np.dot(X, w1.T) + w0
diff = y-y_pred
print(diff.shape)
w0_factors = np.ones((100,1))
w1_update = -(2/100)*0.01*(np.dot(X.T, diff))
w0_update = -(2/100)*0.01*(np.dot(w0_factors.T, diff))   
print(w1_update.shape, w0_update.shape)
w1, w0


# ** 반복적으로 경사 하강법을 이용하여 get_weigth_updates()를 호출하여 w1과 w0를 업데이트 하는 함수 생성 **

# In[ ]:


# 입력 인자 iters로 주어진 횟수만큼 반복적으로 w1과 w0를 업데이트 적용함. 
def gradient_descent_steps(X, y, iters=10000):
    # w0와 w1을 모두 0으로 초기화. 
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    
    # 인자로 주어진 iters 만큼 반복적으로 get_weight_updates() 호출하여 w1, w0 업데이트 수행. 
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01) # Δw1, Δw2 받아옴
        w1 = w1 - w1_update # w1 = w1 - Δw1
        w0 = w0 - w0_update # w0 = w0 - Δw0
        '''
        모든 가중치 (여기선 w1, w0 두 개)를 개별적으로 업데이트하는 게 아니라 iters(epoch)마다 한번에. 
        퍼셉트론 보다는 '아달린'에 가까움!!
        '''
    return w1, w0 # 최종 가중치 리턴 


# ** 예측 오차 비용을 계산을 수행하는 함수 생성 및 경사 하강법 수행 **

# In[ ]:


def get_cost(y, y_pred): #비용함수 구하는 함수. 우리가 줄여야 하는 타겟(비용). 
    N = len(y) 
    cost = np.sum(np.square(y - y_pred))/N  #비용함수 J(w) = 1/2 * Σ( y(i) - y^(i) )^2 를 나타낸 코드. square() : 제곱
    return cost

w1, w0 = gradient_descent_steps(X, y, iters=1000) #최종 업데이트 끝난 가중치 받아옴. 
print("w1:{0:.3f} w0:{1:.3f}".format(w1[0,0], w0[0,0]))
y_pred = w1[0,0] * X + w0  #회귀 계수 : 가중치. x축-X, y축-y_pred인 하나의 1차함수
print('Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))


# In[ ]:


plt.scatter(X, y)
plt.plot(X,y_pred) # X의 feature값이 2개였다면 (column이 2개) -> 3차원의 직선 그래프. 기울기는 얘도 역시 편미분이므로 2개. 


# ** 미니 배치 확률적 경사 하강법을 이용한 최적 비용함수 도출 **

# In[ ]:


def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=1000): ### ◆확률적 경사하강법◆ ###
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 100000
    iter_index =0
    
    for ind in range(iters):
        np.random.seed(ind) # ind라는 규칙(시드값)으로 랜덤하게 뽑겠다. 근데 이 코드 필요 없음. 괜히 쓴듯...ㅎ
        '''
        시드값 하나로 고정 (특정 값으로 정해주거나, 정해주지 않음-default)
            => for문 안에서 여러 번 permutation해도 매번 다른 샘플링
            => for문 돌 때마다 매번 다른 샘플링
            => but, 이 프로젝트 전체를 F5 눌러서 다시 실행 or 전원을 껐다 켜면  >  처음 나온 랜덤값들과 똑같이 나옴. 첫번째 랜덤값끼리, 두번째 랜덤값끼리 ...
        여기처럼 시드값이 계속 바뀜
            => 이 프로젝트 전체를 F5 눌러서 다시 실행 or 전원을 껐다 켜면  >  처음 나온 랜덤값들과 관계 없이, 아예 새로운 랜덤값 나옴.
            
        시드값을 바꾸려면 우리가 직접 코드를 수정해야 한다. 그건 불가능. 
        =====> 보통 어플 출시 전에 시드값으로 date 정보를 넣도록 애초에 코드를 짬. 
        
        '''
        # 전체 X, y 데이터에서 랜덤하게 batch_size만큼 데이터 추출하여 sample_X, sample_y로 저장
        stochastic_random_index = np.random.permutation(X.shape[0]) #np.random.permutation(n) : 0 ~ n-1 숫자가 무작위로 섞인 배열을 만들어준다. epoch마다 랜덤하게 샘플링. 
        sample_X = X[stochastic_random_index[0:batch_size]] #상위에 있는 10개(==batch_size)만 샘플링해서 넘겨준다. 
        sample_y = y[stochastic_random_index[0:batch_size]]
        # 랜덤하게 batch_size만큼 추출된 데이터 기반으로 w1_update, w0_update 계산 후 업데이트
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01) #추출된 데이터에서만 Δw1, Δw2 받아옴. 확률적 경사하강법도 역시 아달린~!
        w1 = w1 - w1_update
        w0 = w0 - w0_update
        '''
        '추출된'(확률적 경사하강법) 모든 가중치 (여기선 w1, w0 두 개)를 개별적으로 업데이트하는 게 아니라 iters(epoch)마다 한번에. 
        이것도 퍼셉트론 보다는 '아달린'에 가까움!!
        '''
    return w1, w0 # 최종 가중치 리턴


# In[ ]:

print(X.shape[0])
test = np.random.permutation(X.shape[0]) 
# np.random.permutation() : 무작위로 섞인 배열을 만들어준다. 
# stochastic 기법 자체가 모든 가중치를 업데이트하지 않음. epoch마다 각각 다르게 샘플링해서, 걔네에 해당하는 가중치들만 업데이트. 


# In[ ]:


w1, w0 = stochastic_gradient_descent_steps(X, y, iters=1000)
print("w1:",round(w1[0,0],3),"w0:",round(w0[0,0],3))
y_pred = w1[0,0] * X + w0
print('Stochastic Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))
'''
퍼셉트론 vs 아달린 vs 확률적 경사하강법(stochastic)
cost : 퍼셉트론 < 아달린 < 확률적 경사하강법
여기서 구한 cost : 0.9935      0.9937     => 비용적으로만 보면 아달린이 더 좋긴 함. 그렇지만 big data 처리할 때 시간이 훨씬 빠름. 

'''
  
plt.scatter(X, y)
plt.plot(X,y_pred) # X의 feature값이 2개였다면 (column이 2개) -> 3차원의 직선 그래프. 기울기는 얘도 역시 편미분이므로 2개. 



############################################################################여기까지함. 



# ## 5.4 사이킷런 LinearRegression을 이용한 보스턴 주택 가격 예측

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # DataFrame형태의 데이터를 그릴 때 seaborn 많이 사용. 
from scipy import stats
from sklearn.datasets import load_boston
get_ipython().run_line_magic('matplotlib', 'inline')

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)
bostonDF.head()


# * CRIM: 지역별 범죄 발생률  
# * ZN: 25,000평방피트를 초과하는 거주 지역의 비율
# * NDUS: 비상업 지역 넓이 비율
# * CHAS: 찰스강에 대한 더미 변수(강의 경계에 위치한 경우는 1, 아니면 0)
# * NOX: 일산화질소 농도
# * RM: 거주할 수 있는 방 개수
# * AGE: 1940년 이전에 건축된 소유 주택의 비율
# * DIS: 5개 주요 고용센터까지의 가중 거리
# * RAD: 고속도로 접근 용이도
# * TAX: 10,000달러당 재산세율
# * PTRATIO: 지역의 교사와 학생 수 비율
# * B: 지역의 흑인 거주 비율
# * LSTAT: 하위 계층의 비율
# * MEDV: 본인 소유의 주택 가격(중앙값)




# * 각 컬럼별로 주택가격에 미치는 영향도를 조사   //각 컬럼별 가중치 구하기 전, 레이블과 무엇이 상관관계가 있는지. 

# In[2]:


# 2개의 행과 4개의 열을 가진 subplots를 이용. axs는 4x2개의 ax를 가짐.
fig, axs = plt.subplots(figsize=(16,8) , ncols=4 , nrows=2) # 행 2개, 컬럼 4개. figsize는 가로 16, 세로 8로 그림판 axs를 크게 만들어달라는 말
lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']
for i , feature in enumerate(lm_features):
    row = int(i/4)  # 0 0 0 0 1 1 1 1 순서대로 들어감
    col = i%4       # 0 1 2 3 0 1 2 3 순서대로 들어감
    
    # 시본의 regplot(선형 회귀 직선 자동적으로 데이터 분포에 의해 알아서 그려줌 ><) 을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature , y='PRICE',data=bostonDF , ax=axs[row][col]) #8개의 각 그래프는 x축을 feature, y축을 'PRICE'로 둠. 
'''
그래프 대충 봤을 때 상관관계 높은 컬럼 
RM: 거주할 수 있는 방 개수  -> 양의 상관관계 높음
LSTAT: 하위 계층의 비율     -> 음의 상관관계 높음
'''


# ** 학습과 테스트 데이터 세트로 분리하고 학습/예측/평가 수행 **

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False) # bostonDF['PRICE']가 실제 label값이므로 얘만 떼어넴!!!

X_train , X_test , y_train , y_test = train_test_split(X_data , y_target ,test_size=0.3, random_state=156)

# Linear Regression OLS로 학습/예측/평가 수행. 
lr = LinearRegression()
lr.fit(X_train ,y_train )
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse) # np.sqrt(n) : √n (루트 씌워줌)

print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))
'''
MSE : 17.297 , RMSE : 4.159
Variance score : 0.757
'''


# In[4]:

# 회귀라는 관점에서 결국 모델링이라는 건, 각 feature(지금 마지막 하나 떼어냈으니까 13가지 컬럼에 곱할 가중치 w13, w12, ..., w1와 w0를 구하는 것! 즉, 13차원의 ~이런 기울기, ~이런 가중치를 갖는 그래프를 그리는 것!
print('절편 값:',lr.intercept_)
print('회귀 계수값:', np.round(lr.coef_, 1)) # lr.coef_를 소수점 첫째 자리까지 반올림해라 => 회귀계수(regression coefficient). 지금 이게 W값 구한 것. 
'''
절편 값: 40.995595172164336
회귀 계수값: [ -0.1   0.1   0.    3.  -19.8   3.4   0.   -1.7   0.4  -0.   -0.9   0.  -0.6]
                                      △ NOX컬럼은 전부 소수점으로, 애초에 데이터값 자체가 다른 컬럼에 비해 상대적으로 많이 작음. 
                                      지금은 전체 피처들을 SCALE해주지 않았기 때문에 얘만 이렇게 크게 나옴. 얘랑 그래프랑 비교하면 굳이 이렇게 높을 이유가 없음. 
                                      스케일링 (SCALE)이 필요하다..!
                                      그래도, 어쨌든 계산된 회귀 계수는 이렇게 하는 게 맞음. 
'''


# In[5]:


# 회귀 계수를 큰 값 순으로 정렬하기 위해 Series로 생성. index가 컬럼명에 유의
coeff = pd.Series(data=np.round(lr.coef_, 1), index=X_data.columns )
coeff.sort_values(ascending=False)


# In[6]:


from sklearn.model_selection import cross_val_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)
lr = LinearRegression() #선형회귀 기반 estimator

# cross_val_score( )로 5 Fold 셋으로 MSE 를 구한 뒤 이를 기반으로 다시  RMSE 구함. 
# train데이터만을 사용해서 거기에서 일정의 train데이터, 일정의 검증데이터로 나눠 cv만큼 교차검증. 퍼포먼스의 우수성은 neg_mean_squared_error, 즉 MSE값이 가장 낮은 게 우수. 
neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores) # scoring="neg_mean_squared_error" 로 잡아줬으니, 그냥 MSE 구하려면 다시 -1 곱해줘야 함. 
avg_rmse = np.mean(rmse_scores)

# cross_val_score(scoring="neg_mean_squared_error")로 반환된 값은 모두 음수 
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))
'''
 5 folds 의 개별 Negative MSE scores:  [-12.46 -26.05 -33.07 -80.76 -33.31]
 5 folds 의 개별 RMSE scores :  [3.53 5.1  5.75 8.99 5.77]
 5 folds 의 평균 RMSE : 5.829 
'''


# ## 5-5. 다항회귀 Polynomial Regression과 오버피팅/언더피팅 이해
# ### 다항회귀 Polynomial Regression 이해

# 다항회귀 PolynomialFeatures 클래스로 다항식 변환
# 
# ![](./image02.png)

# In[7]:


from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 다항식으로 변환한 단항식 생성, [[0,1],[2,3]]의 2X2 행렬 생성
X = np.arange(4).reshape(2,2)
print('일차 단항식 계수 feature:\n',X )

# degree = 2 인 2차 다항식으로 변환하기 위해 PolynomialFeatures를 이용하여 변환
poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
print('변환된 2차 다항식 계수 feature:\n', poly_ftr)
'''
    x : [x1, x2]         → [1, x1, x2, x1², x1*x2, x2²]
    w : [w1, w2, w0]     → [w0, w1, w2, w3,    w4, w5] 
'''


# 3차 다항식 결정값을 구하는 함수 polynomial_func(X) 생성. 즉 회귀식은 결정값 y = 1+ 2x_1 + 3x_1^2 + 4x_2^3 

# In[8]:


def polynomial_func(X):
    y = 1 + 2*X[:,0] + 3*X[:,0]**2 + 4*X[:,1]**3 
    return y

X = np.arange(0,4).reshape(2,2)

print('일차 단항식 계수 feature: \n' ,X)
y = polynomial_func(X)
print('삼차 다항식 결정값: \n', y)

# 3 차 다항식 변환 
poly_ftr = PolynomialFeatures(degree=3).fit_transform(X)
print('3차 다항식 계수 feature: \n',poly_ftr)

# Linear Regression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 계수 확인
model = LinearRegression()
model.fit(poly_ftr,y)
print('Polynomial 회귀 계수\n' , np.round(model.coef_, 2))
print('Polynomial 회귀 Shape :', model.coef_.shape)


# 3차 다항식 계수의 피처값과 3차 다항식 결정값으로 학습

# ** 사이킷런 파이프라인(Pipeline)을 이용하여 3차 다항회귀 학습 **  
# 
# 사이킷런의 Pipeline 객체는 Feature 엔지니어링 변환과 모델 학습/예측을 순차적으로 결합해줍니다. 

# In[9]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def polynomial_func(X):
    y = 1 + 2*X[:,0] + 3*X[:,0]**2 + 4*X[:,1]**3 
    return y

# Pipeline 객체로 Streamline 하게 Polynomial Feature변환과 Linear Regression을 연결
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression())])
X = np.arange(4).reshape(2,2)
y = polynomial_func(X)

model = model.fit(X, y)
print('Polynomial 회귀 계수\n', np.round(model.named_steps['linear'].coef_, 2))


# ** 다항 회귀를 이용한 보스턴 주택가격 예측 **

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)


X_train , X_test , y_train , y_test = train_test_split(X_data , y_target ,test_size=0.3, random_state=156)

## Pipeline을 이용하여 PolynomialFeatures 변환과 LinearRegression 적용을 순차적으로 결합. 
p_model = Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                  ('linear', LinearRegression())])

p_model.fit(X_train, y_train)
y_preds = p_model.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)


print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))


# In[13]:


X_train_poly= PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train, y_train)
X_train_poly.shape, X_train.shape


# ### Polynomial Regression 을 이용한 Underfitting, Overfitting 이해

# ** cosine 곡선에 약간의 Noise 변동값을 더하여 실제값 곡선을 만듬 **

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')

# random 값으로 구성된 X값에 대해 Cosine 변환값을 반환. 
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

# X는 0 부터 1까지 30개의 random 값을 순서대로 sampling 한 데이타 입니다.  
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))

# y 값은 cosine 기반의 true_fun() 에서 약간의 Noise 변동값을 더한 값입니다. 
y = true_fun(X) + np.random.randn(n_samples) * 0.1


# In[ ]:


plt.scatter(X, y)


# In[ ]:


plt.figure(figsize=(14, 5))
degrees = [1, 4, 15]

# 다항 회귀의 차수(degree)를 1, 4, 15로 각각 변화시키면서 비교합니다. 
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())
    
    # 개별 degree별로 Polynomial 변환합니다. 
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X.reshape(-1, 1), y)
    
    # 교차 검증으로 다항 회귀를 평가합니다. 
    scores = cross_val_score(pipeline, X.reshape(-1,1), y,scoring="neg_mean_squared_error", cv=10)
    coefficients = pipeline.named_steps['linear_regression'].coef_
    print('\nDegree {0} 회귀 계수는 {1} 입니다.'.format(degrees[i], np.round(coefficients),2))
    print('Degree {0} MSE 는 {1:.2f} 입니다.'.format(degrees[i] , -1*np.mean(scores)))
    
    # 0 부터 1까지 테스트 데이터 세트를 100개로 나눠 예측을 수행합니다. 
    # 테스트 데이터 세트에 회귀 예측을 수행하고 예측 곡선과 실제 곡선을 그려서 비교합니다.  
    X_test = np.linspace(0, 1, 100)
    # 예측값 곡선
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model") 
    # 실제 값 곡선
    plt.plot(X_test, true_fun(X_test), '--', label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    
    plt.xlabel("x"); plt.ylabel("y"); plt.xlim((0, 1)); plt.ylim((-2, 2)); plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))

plt.show()


# ## 5-6. Regularized Linear Models – Ridge, Lasso
# ### Regularized Linear Model - Ridge Regression

# In[ ]:


# 앞의 LinearRegression예제에서 분할한 feature 데이터 셋인 X_data과 Target 데이터 셋인 Y_target 데이터셋을 그대로 이용 
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)


ridge = Ridge(alpha = 10)
neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 3))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores,3))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))


# ** alpha값을 0 , 0.1 , 1 , 10 , 100 으로 변경하면서 RMSE 측정 **

# In[ ]:


# Ridge에 사용될 alpha 파라미터의 값들을 정의
alphas = [0 , 0.1 , 1 , 10 , 100]

# alphas list 값을 iteration하면서 alpha에 따른 평균 rmse 구함.
for alpha in alphas :
    ridge = Ridge(alpha = alpha)
    
    #cross_val_score를 이용하여 5 fold의 평균 RMSE 계산
    neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    print('alpha {0} 일 때 5 folds 의 평균 RMSE : {1:.3f} '.format(alpha,avg_rmse))


# ** 각 alpha에 따른 회귀 계수 값을 시각화. 각 alpha값 별로 plt.subplots로 맷플롯립 축 생성 **

# In[ ]:


# 각 alpha에 따른 회귀 계수 값을 시각화하기 위해 5개의 열로 된 맷플롯립 축 생성  
fig , axs = plt.subplots(figsize=(18,6) , nrows=1 , ncols=5)
# 각 alpha에 따른 회귀 계수 값을 데이터로 저장하기 위한 DataFrame 생성  
coeff_df = pd.DataFrame()

# alphas 리스트 값을 차례로 입력해 회귀 계수 값 시각화 및 데이터 저장. pos는 axis의 위치 지정
for pos , alpha in enumerate(alphas) :
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_data , y_target)
    # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가.  
    coeff = pd.Series(data=ridge.coef_ , index=X_data.columns )
    colname='alpha:'+str(alpha)
    coeff_df[colname] = coeff
    # 막대 그래프로 각 alpha 값에서의 회귀 계수를 시각화. 회귀 계수값이 높은 순으로 표현
    coeff = coeff.sort_values(ascending=False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3,6)
    sns.barplot(x=coeff.values , y=coeff.index, ax=axs[pos])

# for 문 바깥에서 맷플롯립의 show 호출 및 alpha에 따른 피처별 회귀 계수를 DataFrame으로 표시
plt.show()


# ** alpha 값에 따른 컬럼별 회귀계수 출력 **

# In[ ]:


ridge_alphas = [0 , 0.1 , 1 , 10 , 100]
sort_column = 'alpha:'+str(ridge_alphas[0])
coeff_df.sort_values(by=sort_column, ascending=False)


# ### 라쏘 회귀

# In[ ]:


from sklearn.linear_model import Lasso, ElasticNet

# alpha값에 따른 회귀 모델의 폴드 평균 RMSE를 출력하고 회귀 계수값들을 DataFrame으로 반환 
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None, verbose=True):
    coeff_df = pd.DataFrame()
    if verbose : print('####### ', model_name , '#######')
    for param in params:
        if model_name =='Ridge': model = Ridge(alpha=param)
        elif model_name =='Lasso': model = Lasso(alpha=param)
        elif model_name =='ElasticNet': model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, 
                                             y_target_n, scoring="neg_mean_squared_error", cv = 5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print('alpha {0}일 때 5 폴드 세트의 평균 RMSE: {1:.3f} '.format(param, avg_rmse))
        # cross_val_score는 evaluation metric만 반환하므로 모델을 다시 학습하여 회귀 계수 추출
        
        model.fit(X_data , y_target)
        # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가. 
        coeff = pd.Series(data=model.coef_ , index=X_data.columns )
        colname='alpha:'+str(param)
        coeff_df[colname] = coeff
    return coeff_df
# end of get_linear_regre_eval


# In[ ]:


# 라쏘에 사용될 alpha 파라미터의 값들을 정의하고 get_linear_reg_eval() 함수 호출
lasso_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df =get_linear_reg_eval('Lasso', params=lasso_alphas, X_data_n=X_data, y_target_n=y_target)


# In[ ]:


# 반환된 coeff_lasso_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame출력
sort_column = 'alpha:'+str(lasso_alphas[0])
coeff_lasso_df.sort_values(by=sort_column, ascending=False)


# ### 엘라스틱넷 회귀

# In[ ]:


# 엘라스틱넷에 사용될 alpha 파라미터의 값들을 정의하고 get_linear_reg_eval() 함수 호출
# l1_ratio는 0.7로 고정
elastic_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df =get_linear_reg_eval('ElasticNet', params=elastic_alphas,
                                      X_data_n=X_data, y_target_n=y_target)


# In[ ]:


# 반환된 coeff_elastic_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame출력
sort_column = 'alpha:'+str(elastic_alphas[0])
coeff_elastic_df.sort_values(by=sort_column, ascending=False)


# ### 선형 회귀 모델을 위한 데이터 변환

# In[ ]:


print(y_target.shape)
plt.hist(y_target, bins=10)


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# method는 표준 정규 분포 변환(Standard), 최대값/최소값 정규화(MinMax), 로그변환(Log) 결정
# p_degree는 다향식 특성을 추가할 때 적용. p_degree는 2이상 부여하지 않음. 
def get_scaled_data(method='None', p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log':
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(degree=p_degree, 
                                         include_bias=False).fit_transform(scaled_data)
    
    return scaled_data


# In[ ]:


# Ridge의 alpha값을 다르게 적용하고 다양한 데이터 변환방법에 따른 RMSE 추출. 
alphas = [0.1, 1, 10, 100]
#변환 방법은 모두 6개, 원본 그대로, 표준정규분포, 표준정규분포+다항식 특성
# 최대/최소 정규화, 최대/최소 정규화+다항식 특성, 로그변환 
scale_methods=[(None, None), ('Standard', None), ('Standard', 2), 
               ('MinMax', None), ('MinMax', 2), ('Log', None)]
for scale_method in scale_methods:
    X_data_scaled = get_scaled_data(method=scale_method[0], p_degree=scale_method[1], 
                                    input_data=X_data)
    print('\n## 변환 유형:{0}, Polynomial Degree:{1}'.format(scale_method[0], scale_method[1]))
    get_linear_reg_eval('Ridge', params=alphas, X_data_n=X_data_scaled, 
                        y_target_n=y_target, verbose=False)


# In[ ]:



X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(3)
poly.fit_transform(X)


# In[ ]:




