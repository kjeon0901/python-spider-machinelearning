import numpy as np

#### 퍼셉트론 !!!
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
            for xi, target in zip(X, y): # xi = X row 한줄씩, target = y 하나씩 100바퀴 돈다. (xi, target은 늘 같은 row)
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



ppn = Perceptron(eta=0.1, n_iter=10)  # 이제 우리가 정의한 class 사용!

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o') # 아까 fit함수에서 print(self.errors_)로 출력된 '''[1, 3, 3, 2, 1, 0, 0, 0, 0, 0]'''를 그래프로!
plt.xlabel('Epochs')
plt.ylabel('Number of errors')

plt.show() # 10(epoch == n_iter)번의 가중치 업데이트를 진행하면서 100번 中 error가 발생한 횟수. 
           # 중간에 6번째부터 y축이 0이 되었으므로, 이제는 업데이트가 되지 않으므로 끝까지 쭉 0이다. 


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
    '''
    meshgrid(x, y)의 결과
    xx : x 한 row로 두고 그걸 y 크기만큼 아래로 쫙 복사 -> shape : (y크기, x크기)
    yy : y 한 column으로 두고 그걸 x 크기만큼 오른쪽으로 쫙 복사 -> shape : (y크기, x크기)
    '''
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





#### 아달린 !!!

test=[]
test1=[]
test2=[]
class AdalineGD(object):
    global test
    global test1
    global test2
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        #아무것도 안 넣어줬을 땐 이렇지만, 지금은 객체 생성하면서 각각 n_iter=10, eta=0.01, n_iter=10, eta=0.0001 넣어줬음. 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X) #row 하나씩 넣어준 퍼셉트론과 달리 여기선 (100, 2) shape의 행렬 그대로 들어감. 
            test.append(net_input)
            
            output = self.activation(net_input) #활성화함수로 net_input값 들어간다. 
            
            errors = (y - output) # output y^는 np.dot(X, self.w_[1:]) + self.w_[0] 의 결괏값 계속 그대로 가지고 있음. (퍼셉트론 y^ = 1 or -1 과 다르군!)
                                  # dJ/dw = 	J'(w) = - Σ( y(i) - y^(i) ) * xj(i) 에서 y(i) - y^(i)을 나타낸 code!
                                  # y(i) - y^(i)는 개별 요소를 연산하지만, y - output은 벡터를 연산하는 것!!!!
            test1.append(errors)
            
            self.w_[1:] += self.eta * X.T.dot(errors)   # X.T.dot(errors) : X를 전치행렬시키고, 그 아이와 errors의 내적을 구해라!
                                                        # dJ/dw = 	J'(w) = - Σ( y(i) - y^(i) ) * xj(i) 를 나타낸 코드. 
            test2.append(X.T.dot(errors)) #X의 0번 column과 error와의 내적, X의 1번 column과 error와의 내적 결과인 scalar값이 요소 2개로 저장됨. 
            
            self.w_[0] += self.eta * errors.sum() # w0은 X를 곱할 필요가 없음. 그러니까 X와 내적 없이 그냥 전체를 더해주면 됨. 
            cost = (errors**2).sum() / 2.0 # 아달린의 비용함수 J(w) = 1/2 * Σ( y(i) - y^(i) )^2 를 나타낸 코드. 
            self.cost_.append(cost) #for문 끝날 때 append해줘서, epoch 별로 비용함수가 어떻게 변하는지 담음. 
        return self

    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
        '''
        np.dot(X, self.w_[1:]) 를 어떻게 구해야 할까??
        => X의 각각의 row와의 내적을 구한다! => 결괏값의 shape는 (100,)일 것. 
        X           self.w_[1:]     np.dot(X, self.w_[1:]) __내적
        5.1 1.4     w1              5.1*w1 + 1.4*w2
        4.9 1.4     w2              4.9*w1 + 1.4*w2
        4.7 1.3                     4.7*w1 + 1.3*w2
        ...                         ...
        
        np.dot(X, self.w_[1:]) + self.w_[0] 를 어떻게 구해야 할까??
        => np.dot(X, self.w_[1:])로 나온 (100,) 의 요소 각각에 self.w_[0]를 더해준다! => 결괏값의 shape는 (100,)일 것. 
        np.dot(X, self.w_[1:])  self.w_[0]      np.dot(X, self.w_[1:]) + self.w_[0]
        5.1*w1 + 1.4*w2         w0              5.1*w1 + 1.4*w2 + w0
        4.9*w1 + 1.4*w2                         4.9*w1 + 1.4*w2 + w0
        4.7*w1 + 1.3*w2                         4.7*w1 + 1.3*w2 + w0
        ...                                     ...
        
        '''

    def activation(self, X):
        """선형 활성화 계산"""
        return X # 아~무것도 안 함. 아달린에서 쓰이는 선형 활성화 함수가 ∮(wTㆍx) = wTㆍx 이므로. 다른 활성화 함수는 이 절차에서 다른 output 나올 거임!

    def predict(self, X):
        """최종 예측"""
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o') 
# x축은 epoch 개수만큼 1~10, y축은 아까 구한 epoch별 비용을 log10 (log scale)로 차이 줄여서 그래프로 보기 편하게 보여줌. 
# 근데, 결과 그래프 보면 epoch가 늘어날 수록 error가 "증가하는 방향" 으로 발산해버림;; => 문제가 있구나...!
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y) #eta, 즉 learning rate step을 확 줄여버림. 
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
# 얘는 결과 그래프 보면 epoch가 늘어날 수록 error가 "감소하는 방향" => Good~~~ :)
# 아까는 learning rate step 값이 너무 커서 발산하고 막 튕겨 버렸구나!
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
'''
아달린이 퍼셉트론보다 빠름! 
퍼셉트론은 이중 for문으로, 가중치 업데이트를 너무 일일이 해주기 때문. 
'''