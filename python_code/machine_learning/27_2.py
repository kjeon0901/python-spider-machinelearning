from sklearn.datasets import load_iris  # 붓꽃 데이터 세트 예제를 sklearn.datasets에서 아예 제공해줌
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 정확도 구하는 함수 제공해줌

import pandas as pd

"""
꽃의 길이와 폭, 꽃받침의 길이와 폭 데이터를 가지고 지도학습을 한다. 이것으로 총 3개의 붓꽃 품종 중에서 어떤 종인지 맞춰본다. 
"""
# 붓꽃 데이터 세트를 로딩합니다. iris는 객체가 됨
iris = load_iris()

# iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있습니다.
iris_data = iris.data  # feature값 : 4개의 컬럼네임 Sepal length, Sepal width, Petal length, Petal width

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있습니다.
iris_label = iris.target  # target값 : iris_data라는 feature값을 통해 알아낸 품종은 3개 중 무엇인가 (정답값이 있으므로 지도학습!!)
print("iris target값:", iris_label)
print("iris target명:", iris.target_names)

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
# test=iris_data #얘는 ndarray
# test_=iris.feature_names #얘는 리스트. 리스트든, 딕셔너리든 DataFrame으로 감싸면 DataFrame됨
iris_df = pd.DataFrame(
    data=iris_data, columns=iris.feature_names
)  # data:value값에 해당하는 파라미터, columns:컬럼명에 해당하는 파라미터
iris_df["label"] = iris.target
iris_df.head(3)

# 1. 학습/테스트 데이터 세트 분리
X_train, X_test, y_train, y_test = train_test_split(
    iris_data, iris_label, test_size=0.2, random_state=11
)
"""
X_train     : train 데이터의 feature데이터
y_train     : train 데이터의 label데이터(뒤에 붙은 정답값)
X_test      : test 데이터의 feature데이터
y_test      : test 데이터의 label데이터(뒤에 붙은 정답값)

test_size       - 총 데이터 150개, 그 중 0.2만큼 test로 사용 => 30개 test, 120개 train
random_state    - test데이터도 랜덤하게 뽑아야 한다. 그 때 쓰이는 시드값 : random_state. 
                즉, random_state 설정해주면 매번 뽑히는 X_train, X_test, y_train, y_test값이 동일할 것!!
cf. 랜덤값도 사실 규칙(시드값)이 있다. 
    import random
    random.seed(100)    #100이라는 규칙을 가지고 실행하세요~
    A = random.random() #A값 계속 동일. random.seed(100) 주석처리하면 A값 계속 달라짐.

"""

# 2. DecisionTreeClassifier 객체 생성     //DecisionTreeClassifier : 트리 기반 알고리즘(시각화가능), dt_clf는 그 알고리즘의 객체이고 생성자로 시드값 넣어줌
dt_clf = DecisionTreeClassifier(random_state=11)

# 3. 학습 수행
dt_clf.fit(X_train, y_train)  # 이제 dt_clf는 학습 완료된 모델

# 4. 학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행.
pred = dt_clf.predict(X_test)  # X_test : test데이터의 feature값. y_test에 이미 정답값(실제 레이블값)이 들어 있다.

# 5. 예측 정확도 확인
print(
    "예측 정확도: {0:.4f}".format(accuracy_score(y_test, pred))
)  # 비교할 두 개 넣어주면 됨. 예측된 레이블값 pred와 비교해서 정확도 (맞은개수/전체개수) 비교
