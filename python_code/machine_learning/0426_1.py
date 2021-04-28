import numpy as np
import pandas as pd

"""
문제 1. 

1. 타이타닉 데이터를 데이터 프레임 형태로 업로드
2. null값 얼마나 있는지 확인
3. column별 데이터 분포 확인
4. Pclass의 value_count 확인
5. Age의 null값 채우되 Sibsp가 같은 나이의 평균으로 채운다 ★
    -Sibsp : 0~7까지 있다면, 동일한 Sibsp를 가진 아이들끼리의 묶음의 Age 평균값으로 채우는 것!

cf. 
.apply(lambda x:func(x)) DataFrame에서 람다함수 쓸 때 그냥 이 포맷으로 써라!
if문이 단 하나라도 들어간다거나 하면 func(x)써주는 게 나음~~
	컬럼.apply(lambda x:func(x)) => x:컬럼 안의 요소 하나하나
	x['Age']의 타입은 알 수 없다 (단일요소의 타입:int일 수도, float일 수도, Object(==str)일 수도...)
	DF.apply(lambda x:func(x), axis=1) => x:DataFrame 안의 로우 하나하나


"""
titanic_df = pd.read_csv("C:/jeon/titanic_train.csv")
print(titanic_df.info())
print(titanic_df.isna().sum())
print(titanic_df.describe())
print(titanic_df["Pclass"].value_counts())


def get_category(x):
    if np.isnan(
        x["Age"]
    ):  # isna()는 pandas, 즉 dataframe, series 둘에서만 사용 가능. 여기서 each_row['Age']는 float기 때문에 사용 못함
        if x["SibSp"] == 0:
            return titanic_df[titanic_df["SibSp"] == 0]["Age"].mean()
        elif x["SibSp"] == 1:
            return titanic_df[titanic_df["SibSp"] == 1]["Age"].mean()
        elif x["SibSp"] == 2:
            return titanic_df[titanic_df["SibSp"] == 2]["Age"].mean()
        elif x["SibSp"] == 3:
            return titanic_df[titanic_df["SibSp"] == 3]["Age"].mean()
        elif x["SibSp"] == 4:
            return titanic_df[titanic_df["SibSp"] == 4]["Age"].mean()
        elif x["SibSp"] == 5:
            return titanic_df[titanic_df["SibSp"] == 5]["Age"].mean()
        elif x["SibSp"] == 8:
            return 8
    else:
        return x["Age"]


titanic_df["Age"] = titanic_df.apply(lambda x: get_category(x), axis=1)
# axis=1 => titanic_df의 row 한 줄을 전부 get_category()함수로 던져준다
# axis=0(default)라면 => 컬럼 하나가 여기로 넘어감

"""
문제 1_다른방법. 
import pandas as pd
import numpy as np

titanic_df = pd.read_csv('C:/jeon/titanic_train.csv')
print('titanic 변수 type:',type(titanic_df))
titanic_df_my = titanic_df

print(titanic_df_my.info())
print(titanic_df_my.describe())

value_counts = titanic_df_my['Pclass'].value_counts()
print(value_counts)

value_counts = titanic_df_my['SibSp'].value_counts()
print(value_counts)

def get_category(row):
    cat = ''
    if np.isnan(row['Age']):
        if row['SibSp'] == 0:
            #cat = 0
            cat = titanic_df_my[titanic_df_my['SibSp'] == 0]['Age'].mean()
        elif row['SibSp'] == 1:
            cat = titanic_df_my[titanic_df_my['SibSp'] == 1]['Age'].mean()
        elif row['SibSp'] == 2:
            cat = titanic_df_my[titanic_df_my['SibSp'] == 2]['Age'].mean()
        elif row['SibSp'] == 3:
            cat = titanic_df_my[titanic_df_my['SibSp'] == 3]['Age'].mean()
        elif row['SibSp'] == 4:
            cat = titanic_df_my[titanic_df_my['SibSp'] == 4]['Age'].mean()
        elif row['SibSp'] == 5:
            cat = titanic_df_my[titanic_df_my['SibSp'] == 5]['Age'].mean()
        elif row['SibSp'] == 8:
            cat = 8   
    else:
        cat = row['Age']

    return int(cat)

titanic_df_my['Age'] = titanic_df_my.apply(lambda x : get_category(x), axis=1)
print(titanic_df_my['Age'].notnull().any())     #'Age'컬럼이 전부 null이 아니어야 True
        #isna(), isnull()   : null(O) - True, null(X) - False
        #notnull()          : null(O) - False, null(X) - True
        #all() : 모두 참이어야 참, 하나라도 거짓이면 거짓
        #any() : 하나라도 참이면 참, 모두 거짓이어야 거짓
"""
