import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])[:, np.newaxis] 
            # np.newaxis : 차원(dimension)을 하나 더 높여줌. (10,) → (10, 1)

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2, 390.8])
            # 얘는 여전히 (10,)

from sklearn.preprocessing import PolynomialFeatures #★★★

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2) # degree=2 : 2차 다항식으로 만들어줄 것. ax+b → ax²+bx+c
X_quad = quadratic.fit_transform(X)
'''
    ax+b → ax²+bx+c
    ..................................
    x : [x]         → [x²,  x,  1]
    w : [w1, w0]    → [w2, w1, w0]      //가중치 4개 아니고 3개인 이유 : [w3, w2, w1, w0]으로 만들어졌는데, 사실상 w1은 계속 1만 곱해지기 때문에 w1, w0 모두 bias. 그래서 그냥 합쳐버림. 
    Y는 일정
    
    y = w1*x + w0   → w2*x² + w1*x + w0. 
    이제 구해진 새로운 2차식 y = w2*x² + w1*x + w0 에서 퍼셉트론/아달린/경사하강법 등등을 사용해 최적의 w 구함.
    => 그래프로 보면, 1차식만 사용해 구하던 걸, 이제는 2차함수를 사용해 MSE값을 구한 것. 
'''


# 선형 특성 학습
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# 이차항 특성 학습
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# 결과 그래프
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()