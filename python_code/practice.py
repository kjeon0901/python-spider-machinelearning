import numpy as np


class Perceptron(object):
    """퍼셉트론 분류기

    매개변수
    ------------
    eta : float
      학습률 learning rate (0.0과 1.0 사이)
      Δwj = η( y(i) - y^(i) ) * xj(i)에서   η
    n_iter : int
      훈련 데이터셋 반복 횟수. 에포크(epoch)값
    random_state : int
      가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    -----------
    w_ : 1d-array
      학습된 가중치
    errors_ : list
      에포크마다 누적된 분류 오류

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1): 
        #아무것도 안 넣어줬을 땐 이렇지만, 지금은 객체 생성하면서 eta=0.1, n_iter=10 넣어줬음. 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """훈련 데이터 학습

        매개변수
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]
          타깃값

        반환값
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # normal(loc=평균, scale=표준편차, size=몇개의 데이터를 뱉어줄까?) : 평균이 loc이고 표준편차가 scale인 정규분포(평균:0 분산:1) 데이터를 size만큼 뱉어 달라!
        print(self.w_) # 3개의 랜덤값으로 가중치 3개 초기화 '''[ 0.01624345 -0.00611756 -0.00528172]'''
        
        self.errors_ = []

        for _ in range(self.n_iter): # 10바퀴만 돈다. 
            errors = 0
            for xi, target in zip(X, y): # xi = X row 한줄씩, target = y 하나씩  (xi, target은 늘 같은 row)
                # 가중치 변화량 Δwj = η( y(i) - y^(i) ) * xj(i)
                update = self.eta * (target - self.predict(xi)) # η( y(i) - y^(i) ) 까지만 구함. xi: X row 하나씩 들어감(vector).
                                                                # target, self.predict(xi), self.eta 모두 scalar => update도 scalar값. 
                self.w_[1:] += update * xi # update는 scalar, xi는 vector. xi 요소 하나하나에 update 곱해줌. (동시에 모든 가중치 업데이트. wj = wj + Δwj)
                self.w_[0] += update # w0은 x값과 곱해주지 않고 x와는 무관하므로 xi 곱해주면 안됨!
                errors += int(update != 0.0) # y != y^인 경우 update != 0.0. 두 번째 for문 안에서 100번 도는 중에, 몇 번 잘못 예측했는가 cnt. 
            self.errors_.append(errors)
        print(self.errors_)
        return self

    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0] # np.dot(A, B) → A와 B 행렬 내적 (==행렬 곱)
        # z = wTㆍx에서 w : w_[1:], x : X
        # self.w_[0] : 편차. 상수값. 해당 그래프를 얼마만큼 shift시키는지. ax+b에서 b, ax^2+bx+c에서 c. 보통 이것도 같이 설정해준다. 
        # x1*w1 + x2*w2 + w0 (w0 : x와는 무관. Deep learning에는 나중에 bias(얼마나 원점에서 shift되는가)라는 개념이 나오는데, 그것과 비슷한 개념)
        
    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다"""
        return np.where(self.net_input(X) >= 0.0, 1, -1) # z값 (self.net_input(X)) 구해서 단위계단함수 ∮(z) = z≥0일때 1이고 아니면 -1인 함수 만듦. 리턴되는 값은 예측값 ∮(z)



v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()


%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# setosa와 versicolor를 선택합니다
y = df.iloc[0:100, 4].values # 총 150개 중 100개 데이터만, 4번째 column(붓꽃데이터 레이블값 str 그대로 받아옴)
y = np.where(y == 'Iris-setosa', -1, 1) # y에 'Iris-setosa'라면 -1, 그게 아니면 1을 담아라.

# 꽃받침 길이와 꽃잎 길이를 추출합니다
X = df.iloc[0:100, [0, 2]].values # 총 150개 중 100개 데이터만, 컬럼(feature)은 0번째, 2번째만.

# scatter(): 점 찍는 그래프를 그립니다
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show() #그리기~

###### 이제 우리가 정의한 class 사용
ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of errors')

plt.show()


from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # 마커와 컬러맵을 설정합니다
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 샘플의 산점도를 그립니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()