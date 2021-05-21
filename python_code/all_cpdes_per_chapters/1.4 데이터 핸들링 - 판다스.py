#!/usr/bin/env python
# coding: utf-8

# ### Pandas 시작- 파일을 DataFrame 로딩, 기본 API

# In[3]:


import pandas as pd


# In[4]:


titanic_df = pd.read_csv('titanic_train.csv')
print('titanic 변수 type:',type(titanic_df))


# **read_csv()**  
# read_csv()를 이용하여 csv 파일을 편리하게 DataFrame으로 로딩합니다.   
# read_csv() 의 sep 인자를 콤마(,)가 아닌 다른 분리자로 변경하여 다른 유형의 파일도 로드가 가능합니다.

# **head()**  
# DataFrame의 맨 앞 일부 데이터만 추출합니다.

# In[5]:


titanic_df.head(5)


# ** DataFrame의 생성 **

# In[4]:


dic1 = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
# 딕셔너리를 DataFrame으로 변환
data_df = pd.DataFrame(dic1)
print(data_df)
print("#"*30)

# 새로운 컬럼명을 추가
data_df = pd.DataFrame(dic1, columns=["Name", "Year", "Gender", "Age"])
print(data_df)
print("#"*30)

# 인덱스를 새로운 값으로 할당. 
data_df = pd.DataFrame(dic1, index=['one','two','three','four'])
print(data_df)
print("#"*30)


# **DataFrame의 컬럼명과 인덱스**

# In[6]:


print("columns:",titanic_df.columns)
print("index:",titanic_df.index)
print("index value:", titanic_df.index.values)


# **DataFrame에서 Series 추출 및 DataFrame 필터링 추출**

# In[6]:


# DataFrame객체에서 []연산자내에 한개의 컬럼만 입력하면 Series 객체를 반환  
series = titanic_df['Name']
print(series.head(3))
print("## type:",type(series))

# DataFrame객체에서 []연산자내에 여러개의 컬럼을 리스트로 입력하면 그 컬럼들로 구성된 DataFrame 반환  
filtered_df = titanic_df[['Name', 'Age']]
print(filtered_df.head(3))
print("## type:", type(filtered_df))

# DataFrame객체에서 []연산자내에 한개의 컬럼을 리스트로 입력하면 한개의 컬럼으로 구성된 DataFrame 반환 
one_col_df = titanic_df[['Name']]
print(one_col_df.head(3))
print("## type:", type(one_col_df))


# ** shape **  
# DataFrame의 행(Row)와 열(Column) 크기를 가지고 있는 속성입니다.

# In[7]:


print('DataFrame 크기: ', titanic_df.shape)


# **info()**  
# DataFrame내의 컬럼명, 데이터 타입, Null건수, 데이터 건수 정보를 제공합니다. 

# In[8]:


titanic_df.info()


# **describe()**  
# 데이터값들의 평균,표준편차,4분위 분포도를 제공합니다. 숫자형 컬럼들에 대해서 해당 정보를 제공합니다.

# In[9]:


titanic_df.describe()


# **value_counts()**  
# 동일한 개별 데이터 값이 몇건이 있는지 정보를 제공합니다. 즉 개별 데이터값의 분포도를 제공합니다. 
# 주의할 점은 value_counts()는 Series객체에서만 호출 될 수 있으므로 반드시 DataFrame을 단일 컬럼으로 입력하여 Series로 변환한 뒤 호출합니다. 

# In[10]:


value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)


# In[4]:


titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))


# In[12]:


titanic_pclass.head()


# In[13]:


value_counts = titanic_df['Pclass'].value_counts()
print(type(value_counts))
print(value_counts)


# **sort_values()**
# by=정렬컬럼, ascending=True 또는 False로 오름차순/내림차순으로 정렬

# In[14]:


titanic_df.sort_values(by='Pclass', ascending=True)

titanic_df[['Name','Age']].sort_values(by='Age')
titanic_df[['Name','Age','Pclass']].sort_values(by=['Pclass','Age'])


# #### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환
# 
# **리스트, ndarray에서 DataFrame변환**

# In[7]:


import numpy as np

col_name1=['col1']
list1 = [1, 2, 3]
array1 = np.array(list1)

print('array1 shape:', array1.shape )
df_list1 = pd.DataFrame(list1, columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
df_array1 = pd.DataFrame(array1, columns=col_name1)
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)


# In[8]:


# 3개의 컬럼명이 필요함. 
col_name2=['col1', 'col2', 'col3']

# 2행x3열 형태의 리스트와 ndarray 생성 한 뒤 이를 DataFrame으로 변환. 
list2 = [[1, 2, 3],
         [11, 12, 13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape )
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
df_array1 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array1)


# ** 딕셔너리(dict)에서 DataFrame변환**

# In[17]:


# Key는 컬럼명으로 매핑, Value는 리스트 형(또는 ndarray)
dict = {'col1':[1, 11], 'col2':[2, 22], 'col3':[3, 33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n', df_dict)


# ** DataFrame을 ndarray로 변환**

# In[18]:


# DataFrame을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.values shape:', array3.shape)
print(array3)


# **DataFrame을 리스트와 딕셔너리로 변환**

# In[19]:


# DataFrame을 리스트로 변환
list3 = df_dict.values.tolist()
print('df_dict.values.tolist() 타입:', type(list3))
print(list3)

# DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('\n df_dict.to_dict() 타입:', type(dict3))
print(dict3)


# ### DataFrame의 컬럼 데이터 셋 Access
# 
# DataFrame의 컬럼 데이터 세트 생성과 수정은 [ ] 연산자를 이용해 쉽게 할 수 있습니다. 
# 새로운 컬럼에 값을 할당하려면 DataFrame [ ] 내에 새로운 컬럼명을 입력하고 값을 할당해주기만 하면 됩니다

# In[14]:


titanic_df['Age_0']=0
titanic_df.head(3)


# In[15]:


titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
titanic_df.head(3)


# 기존 컬럼에 값을 업데이트 하려면 해당 컬럼에 업데이트값을 그대로 지정하면 됩니다.

# In[16]:


titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
titanic_df.head(3)


# ### DataFrame 데이터 삭제
# 
# ** axis에 따른 삭제**

# In[17]:


titanic_drop_df = titanic_df.drop('Age_0', axis=1 )
titanic_drop_df.head(3)


# drop( )메소드의 inplace인자의 기본값은 False 입니다.
# 이 경우 drop( )호출을 한 DataFrame은 아무런 영향이 없으며 drop( )호출의 결과가 해당 컬럼이 drop 된 DataFrame을 반환합니다. 

# In[12]:


titanic_df.head(3)


# 여러개의 컬럼들의 삭제는 drop의 인자로 삭제 컬럼들을 리스트로 입력합니다. 
# inplace=True 일 경우 호출을 한 DataFrame에 drop이 반영됩니다. 이 때 반환값은 None입니다.

# In[18]:


drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
print(' inplace=True 로 drop 후 반환된 값:',drop_result)
titanic_df.head(3)


# axis=0 일 경우 drop()은 row 방향으로 데이터를 삭제합니다. 

# In[26]:


titanic_df = pd.read_csv('titanic_train.csv')
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('#### before axis 0 drop ####')
print(titanic_df.head(6))

titanic_df.drop([0,1,2], axis=0, inplace=True)

print('#### after axis 0 drop ####')
print(titanic_df.head(3))


# ### Index 객체

# In[27]:


# 원본 파일 재 로딩 
titanic_df = pd.read_csv('titanic_train.csv')
# Index 객체 추출
indexes = titanic_df.index
print(indexes)
# Index 객체를 실제 값 arrray로 변환 
print('Index 객체 array값:\n',indexes.values)


# Index는 1차원 데이터 입니다.

# In[28]:


print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])


# [ ]를 이용하여 임의로 Index의 값을 변경할 수는 없습니다. 

# In[29]:


indexes[0] = 5


# Series 객체는 Index 객체를 포함하지만 Series 객체에 연산 함수를 적용할 때 Index는 연산에서 제외됩니다. Index는 오직 식별용으로만 사용됩니다.

# In[30]:


series_fair = titanic_df['Fare']
series_fair.head(5)


# In[31]:


print('Fair Series max 값:', series_fair.max())
print('Fair Series sum 값:', series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n',(series_fair + 3).head(3) )


# DataFrame 및 Series에 reset_index( ) 메서드를 수행하면 새롭게 인덱스를 연속 숫자 형으로 할당하며 기존 인덱스는 ‘index’라는 새로운 컬럼 명으로 추가합니다

# In[32]:


titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)


# In[33]:


titanic_reset_df.shape


# In[34]:


print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입:',type(value_counts))

new_value_counts = value_counts.reset_index(inplace=False)
print('### After reset_index ###')
print(new_value_counts)
print('new_value_counts 객체 변수 타입:',type(new_value_counts))


# ### 데이터 Selection 및 Filtering

# **DataFrame의 [ ] 연산자**  
# 
# 넘파이에서 [ ] 연산자는 행의 위치, 열의 위치, 슬라이싱 범위 등을 지정해 데이터를 가져올 수 있었습니다.
# 하지만 DataFrame 바로 뒤에 있는 ‘[ ]’ 안에 들어갈 수 있는 것은 컬럼 명 문자(또는 컬럼 명의 리스트
# 객체), 또는 인덱스로 변환 가능한 표현식입니다. 

# In[35]:


titanic_df = pd.read_csv('titanic_train.csv')
print('단일 컬럼 데이터 추출:\n', titanic_df[ 'Pclass' ].head(3))
print('\n여러 컬럼들의 데이터 추출:\n', titanic_df[ ['Survived', 'Pclass'] ].head(3))
print('[ ] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])


# 앞에서 DataFrame의 [ ] 내에 숫자 값을 입력할 경우 오류가 발생한다고 했는데, Pandas의 Index 형태로 변환가능한    
# 표현식은 [ ] 내에 입력할 수 있습니다.  
# 가령 titanic_df의 처음 2개 데이터를 추출하고자  titanic_df [ 0:2 ] 와 같은 슬라이싱을 이용하였다면 정확히 원하는 결과를 반환해 줍니다. 

# In[36]:


titanic_df[0:2]


# [ ] 내에 조건식을 입력하여 불린 인덱싱을 수행할 수 있습니다(DataFrame 바로 뒤에 있는 []안에 들어갈 수 있는 것은 컬럼명과 불린인덱싱으로 범위를 좁혀서 코딩을 하는게 도움이 됩니다)

# In[37]:


titanic_df[ titanic_df['Pclass'] == 3].head(3)


# **DataFrame ix[] 연산자**  
# 명칭 기반과 위치 기반 인덱싱 모두를 제공. 

# In[38]:


titanic_df.head(3)


# In[39]:


print('컬럼 위치 기반 인덱싱 데이터 추출:',titanic_df.ix[0,2])
print('컬럼명 기반 인덱싱 데이터 추출:',titanic_df.ix[0,'Pclass'])


# In[40]:


data = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
data_df = pd.DataFrame(data, index=['one','two','three','four'])
data_df


# In[41]:


print("\n ix[0,0]", data_df.ix[0,0])
print("\n ix['one', 0]", data_df.ix['one',0])
print("\n ix[3, 'Name']",data_df.ix[3, 'Name'],"\n")

print("\n ix[0:2, [0,1]]\n", data_df.ix[0:2, [0,1]])
print("\n ix[0:2, [0:3]]\n", data_df.ix[0:2, 0:3])
print("\n ix[0:3, ['Name', 'Year']]\n", data_df.ix[0:3, ['Name', 'Year']], "\n")
print("\n ix[:] \n", data_df.ix[:])
print("\n ix[:, :] \n", data_df.ix[:, :])

print("\n ix[data_df.Year >= 2014] \n", data_df.ix[data_df.Year >= 2014])


# In[42]:


# data_df 를 reset_index() 로 새로운 숫자형 인덱스를 생성
data_df_reset = data_df.reset_index()
data_df_reset = data_df_reset.rename(columns={'index':'old_index'})

# index 값에 1을 더해서 1부터 시작하는 새로운 index값 생성
data_df_reset.index = data_df_reset.index+1
data_df_reset


# In[43]:


# 아래 코드는 오류를 발생합니다. 
data_df_reset.ix[0,1]


# In[44]:


data_df_reset.ix[1,1]


# ** DataFrame iloc[ ] 연산자**  
# 위치기반 인덱싱을 제공합니다.

# In[45]:


data_df.head()


# In[46]:


data_df.iloc[0, 0]


# In[47]:


# 아래 코드는 오류를 발생합니다. 
data_df.iloc[0, 'Name']


# In[48]:


# 아래 코드는 오류를 발생합니다. 
data_df.iloc['one', 0]


# In[49]:


data_df_reset.head()


# In[50]:


data_df_reset.iloc[0, 1]


# **DataFrame loc[ ] 연산자**  
# 명칭기반 인덱싱을 제공합니다.

# In[51]:


data_df


# In[52]:


data_df.loc['one', 'Name']


# In[53]:


data_df_reset.loc[1, 'Name']


# In[54]:


# 아래 코드는 오류를 발생합니다. 
data_df_reset.loc[0, 'Name']


# In[55]:


print('명칭기반 ix slicing\n', data_df.ix['one':'two', 'Name'],'\n')
print('위치기반 iloc slicing\n', data_df.iloc[0:1, 0],'\n')
print('명칭기반 loc slicing\n', data_df.loc['one':'two', 'Name'])


# In[68]:


print(data_df_reset.loc[1:2 , 'Name'])


# In[69]:


print(data_df.ix[1:2 , 'Name'])


# ** 불린 인덱싱(Boolean indexing) **  
# 헷갈리는 위치기반, 명칭기반 인덱싱을 사용할 필요없이 조건식을 [ ] 안에 기입하여 간편하게 필터링을 수행.

# In[ ]:


titanic_df = pd.read_csv('titanic_train.csv')
titanic_boolean = titanic_df[titanic_df['Age'] > 60]
print(type(titanic_boolean))
titanic_boolean


# In[ ]:


titanic_df['Age'] > 60

var1 = titanic_df['Age'] > 60
print(type(var1))


# In[ ]:


titanic_df[titanic_df['Age'] > 60][['Name','Age']].head(3)


# In[ ]:


titanic_df[['Name','Age']][titanic_df['Age'] > 60].head(3)


# In[ ]:


titanic_df.loc[titanic_df['Age'] > 60, ['Name','Age']].head(3)


# 논리 연산자로 결합된 조건식도 불린 인덱싱으로 적용 가능합니다. 

# In[ ]:


titanic_df[ (titanic_df['Age'] > 60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')]


# 조건식은 변수로도 할당 가능합니다. 복잡한 조건식은 변수로 할당하여 가득성을 향상 할 수 있습니다.

# In[ ]:


cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
titanic_df[ cond1 & cond2 & cond3]


# In[ ]:


import pandas as pd
titanic_df = pd.read_csv('titanic_train.csv')


# ### Aggregation 함수 및 GroupBy 적용
# ** Aggregation 함수 **

# In[19]:


## NaN 값은 count에서 제외
titanic_df.count()


# 특정 컬럼들로 Aggregation 함수 수행.

# In[20]:


titanic_df[['Age', 'Fare']].mean(axis=1)


# In[21]:


titanic_df[['Age', 'Fare']].sum(axis=0)


# In[22]:


titanic_df[['Age', 'Fare']].count()


# **groupby( )**
# by 인자에 Group By 하고자 하는 컬럼을 입력, 여러개의 컬럼으로 Group by 하고자 하면 [ ] 내에 해당 컬럼명을 입력. DataFrame에 groupby( )를 호출하면 DataFrameGroupBy 객체를 반환. 

# In[7]:


titanic_groupby = titanic_df.groupby(by='Pclass')
print(type(titanic_groupby)) 
print(titanic_groupby)


# In[8]:


DataFrameGroupBy객체에 Aggregation함수를 호출하여 Group by 수행.


# In[9]:


titanic_groupby = titanic_df.groupby('Pclass').count()
titanic_groupby


# In[ ]:


print(type(titanic_groupby))
print(titanic_groupby.shape)
print(titanic_groupby.index)


# In[ ]:


titanic_groupby = titanic_df.groupby(by='Pclass')[['PassengerId', 'Survived']].count()
titanic_groupby


# In[ ]:


titanic_df[['Pclass','PassengerId', 'Survived']].groupby('Pclass').count()


# In[ ]:


titanic_df.groupby('Pclass')['Pclass'].count()
titanic_df['Pclass'].value_counts()


# RDBMS의 group by는 select 절에 여러개의 aggregation 함수를 적용할 수 있음. 
# 
# Select max(Age), min(Age) from titanic_table group by Pclass
# 
# 판다스는 여러개의 aggregation 함수를 적용할 수 있도록 agg( )함수를 별도로 제공

# In[ ]:


titanic_df.groupby('Pclass')['Age'].agg([max, min])


# 딕셔너리를 이용하여 다양한 aggregation 함수를 적용
# 

# In[ ]:


agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)


# ### Missing 데이터 처리하기
# DataFrame의 isna( ) 메소드는 모든 컬럼값들이 NaN인지 True/False값을 반환합니다(NaN이면 True)

# In[ ]:


titanic_df.isna().head(3)


# 아래와 같이 isna( ) 반환 결과에 sum( )을 호출하여 컬럼별로 NaN 건수를 구할 수 있습니다. 

# In[ ]:


titanic_df.isna( ).sum( )


# ** fillna( ) 로 Missing 데이터 대체하기 **

# In[ ]:


titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
titanic_df.head(3)


# In[ ]:


titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()


# ### apply lambda 식으로 데이터 가공
# 파이썬 lambda 식 기본

# In[73]:


def get_square(a):
    return a**2

print('3의 제곱은:',get_square(3))


# In[74]:


lambda_square = lambda x : x ** 2
print('3의 제곱은:',lambda_square(3))


# In[75]:


a=[1,2,3]
squares = map(lambda x : x**2, a)
list(squares)


# ** 판다스에 apply lambda 식 적용 **

# In[76]:


titanic_df['Name_len']= titanic_df['Name'].apply(lambda x : len(x))
titanic_df[['Name','Name_len']].head(3)


# In[77]:


titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult' )
titanic_df[['Age','Child_Adult']].head(10)


# In[78]:


titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x <= 60 else 
                                                                                  'Elderly'))
titanic_df['Age_cat'].value_counts()


# In[79]:


def get_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
titanic_df[['Age','Age_cat']].head()
    


# In[ ]:




