from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
kfold = KFold(n_splits=5)
cv_accuracy = []
print("붓꽃 데이터 세트 크기:", features.shape[0])

n_iter = 0
train_index_chk = []
test_index_chk = []

# KFold객체의 split( ) 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(
    features
):  # train_index, test_index (kfold.split(features)로 리턴)로 학습용/검증용 데이터 추출해야 하는구나~
    # 원래는 kfold.split(A)에서 A 안에 train_test_split()한 뒤 train데이터만 넣어야 하는데, 그냥 전체 데이터 features를 train데이터라 생각하고 넣어보았다~
    train_index_chk.append(train_index)  # variable explorer 보면 ndarray 하나씩 5번 들어감. for문이 5번 돌았구나~
    test_index_chk.append(test_index)  # variable explorer 보면 ndarray 하나씩 5번 들어감. for문이 5번 돌았구나~

    # kfold.split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    # 반복 시 마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred), 4)  # np.round(a, b) : a를 소수점 4째자리까지 반올림
    train_size = X_train.shape[0]  # 120    ← X_train.shape : (120, 4)
    test_size = X_test.shape[0]  # 30     ← X_test.shape  : (30, 4)
    print(
        "\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}".format(
            n_iter, accuracy, train_size, test_size
        )
    )
    print("#{0} 검증 세트 인덱스:{1}".format(n_iter, test_index))
    cv_accuracy.append(accuracy)  # 리스트로 담음

# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print("\n## 평균 검증 정확도:", np.mean(cv_accuracy))
