#판다스 맛보기 - kaggle - titanic
import csv
import pandas as pd

df_titanic = pd.read_csv("C:/jeon/titanic_train.csv")
'''
df-titanic의 타입: DataFrame, 사이즈:(891, 12) 즉 행렬!, column명과 index가 하나로 묶여 있는 데이터
DataFrame이라는 데이터 타입 안에 Pandas라는 패키지가 있다
Pandas : DataFrame 타입의 데이터를 굉장히 잘 다룰 수 있게 해줌
'''

print(df_titanic.info())
'''
↓출력
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype     #모델링한다 = 가중치를 찾아 낸다. 거기서 null값은 사용할 수 없으니까 채워 넣으라고 Non-Null Count 보여줌. 
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
'''

print(df_titanic.describe())
'''
↓출력 (표준편차: 분포 보여줌)
       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200
'''
