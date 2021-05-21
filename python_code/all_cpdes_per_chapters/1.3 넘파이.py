#!/usr/bin/env python
# coding: utf-8

# ### Numpy ndarray 개요

# * ndarray 생성
# np.array()

# In[ ]:


import numpy as np


# In[ ]:


list1 = [1, 2, 3]
print("list1:",list1)
print("list1 type:",type(list1))

array1 = np.array(list1)
print("array1:",array1)
print("array1 type:", type(array1))


# * ndarray 의 형태(shape)와 차원

# In[ ]:


array1 = np.array([1,2,3])
print('array1 type:',type(array1))
print('array1 array 형태:',array1.shape)

array2 = np.array([[1,2,3],
                  [2,3,4]])
print('array2 type:',type(array2))
print('array2 array 형태:',array2.shape)

array3 = np.array([[1,2,3]])
print('array3 type:',type(array3))
print('array3 array 형태:',array3.shape)


# In[ ]:


print('array1: {:0}차원, array2: {:1}차원, array3: {:2}차원'.format(array1.ndim,array2.ndim,array3.ndim))


# * ndarray 데이터 값 타입

# In[ ]:


list1 = [1,2,3]
print(type(list1))
array1 = np.array(list1)

print(type(array1))
print(array1, array1.dtype)


# In[ ]:


list2 = [1, 2, 'test']
array2 = np.array(list2)
print(array2, array2.dtype)

list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)


# * astype()을 통한 타입 변환

# In[ ]:


array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

array_int1= array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2= array_float1.astype('int32')
print(array_int2, array_int2.dtype)


# * ndarray에서 axis 기반의 연산함수 수행

# In[ ]:


array2 = np.array([[1,2,3],
                  [2,3,4]])

print(array2.sum())
print(array2.sum(axis=0))
print(array2.sum(axis=1))


# * ndarray를 편리하게 생성하기 - arange, zeros, ones

# In[ ]:


sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)


# In[ ]:


zero_array = np.zeros((3,2),dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)

one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)


# * ndarray의 shape를 변경하는 reshape()

# In[ ]:


array1 = np.arange(10)
print('array1:\n', array1)

array2 = array1.reshape(2,5)
print('array2:\n',array2)

array3 = array1.reshape(5,2)
print('array3:\n',array3)


# In[ ]:


# 변환할 수 있는 shape구조를 입력하면 오류 발생.
array1.reshape(4,3)


# reshape()에 -1 인자값을 부여하여 특정 차원으로 고정된 가변적인 ndarray형태 변환

# In[ ]:


array1 = np.arange(10)
print(array1)

#컬럼 axis 크기는 5에 고정하고 로우 axis크기를 이에 맞춰 자동으로 변환. 즉 2x5 형태로 변환 
array2 = array1.reshape(-1,5)
print('array2 shape:',array2.shape)
print('array2:\n', array2)

#로우 axis 크기는 5로 고정하고 컬럼 axis크기는 이에 맞춰 자동으로 변환. 즉 5x2 형태로 변환 
array3 = array1.reshape(5,-1)
print('array3 shape:',array3.shape)
print('array3:\n', array3)


# In[ ]:


# reshape()는 (-1, 1), (-1,)와 같은 형태로 주로 사용됨.
# 1차원 ndarray를 2차원으로 또는 2차원 ndarray를 1차원으로 변환 시 사용. 
array1 = np.arange(5)

# 1차원 ndarray를 2차원으로 변환하되, 컬럼axis크기는 반드시 1이여야 함. 
array2d_1 = array1.reshape(-1, 1)
print("array2d_1 shape:", array2d_1.shape)
print("array2d_1:\n",array2d_1)

# 2차원 ndarray를 1차원으로 변환 
array1d = array2d_1.reshape(-1,)
print("array1d shape:", array1d.shape)
print("array1d:\n",array1d)


# In[ ]:


# -1 을 적용하여도 변환이 불가능한 형태로의 변환을 요구할 경우 오류 발생.
array1 = np.arange(10)
array4 = array1.reshape(-1,4)


# In[ ]:


# 반드시 -1 값은 1개의 인자만 입력해야 함. 
array1.reshape(-1, -1)


# ### ndarray의 데이터 세트 선택하기 – 인덱싱(Indexing)

# **특정 위치의 단일값 추출**

# In[45]:


# 1에서 부터 9 까지의 1차원 ndarray 생성 
array1 = np.arange(start=1, stop=10)
print('array1:',array1)

# index는 0 부터 시작하므로 array1[2]는 3번째 index 위치의 데이터 값을 의미
value = array1[2]
print('value:',value)
print(type(value))


# In[46]:


print('맨 뒤의 값:',array1[-1], ', 맨 뒤에서 두번째 값:',array1[-2])


# In[47]:


array1[0] = 9
array1[8] = 0
print('array1:',array1)


# In[48]:


array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)

print('(row=0,col=0) index 가리키는 값:', array2d[0,0] )
print('(row=0,col=1) index 가리키는 값:', array2d[0,1] )
print('(row=1,col=0) index 가리키는 값:', array2d[1,0] )
print('(row=2,col=2) index 가리키는 값:', array2d[2,2] )


# **슬라이싱(Slicing)**

# In[49]:


array1 = np.arange(start=1, stop=10)
print(array1)
array3 = array1[0:3]
print(array3)
print(type(array3))


# In[50]:


array1 = np.arange(start=1, stop=10)
array4 = array1[:3]
print(array4)

array5 = array1[3:]
print(array5)

array6 = array1[:]
print(array6)


# In[51]:


array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print('array2d:\n',array2d)

print('array2d[0:2, 0:2] \n', array2d[0:2, 0:2])
print('array2d[1:3, 0:3] \n', array2d[1:3, 0:3])
print('array2d[1:3, :] \n', array2d[1:3, :])
print('array2d[:, :] \n', array2d[:, :])
print('array2d[:2, 1:] \n', array2d[:2, 1:])
print('array2d[:2, 0] \n', array2d[:2, 0])


# ** 팬시 인덱싱(fancy indexing) **

# In[52]:


array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)

array3 = array2d[[0,1], 2]
print('array2d[[0,1], 2] => ',array3.tolist())

array4 = array2d[[0,2], 0:2]
print('array2d[[0,2], 0:2] => ',array4.tolist())

array5 = array2d[[0,1]]
print('array2d[[0,1]] => ',array5.tolist())


# ** 불린 인덱싱(Boolean indexing) **

# In[53]:


array1d = np.arange(start=1, stop=10)
print(array1d)


# In[55]:


print(array1d > 5)

var1 = array1d > 5
print("var1:",var1)
print(type(var1))


# In[56]:


# [ ] 안에 array1d > 5 Boolean indexing을 적용 
print(array1d)
array3 = array1d[array1d > 5]
print('array1d > 5 불린 인덱싱 결과 값 :', array3)


# In[57]:


boolean_indexes = np.array([False, False, False, False, False,  True,  True,  True,  True])
array3 = array1d[boolean_indexes]
print('불린 인덱스로 필터링 결과 :', array3)


# In[58]:


indexes = np.array([5,6,7,8])
array4 = array1d[ indexes ]
print('일반 인덱스로 필터링 결과 :',array4)


# In[59]:


array1d = np.arange(start=1, stop=10)
target = []

for i in range(0, 9):
    if array1d[i] > 5:
        target.append(array1d[i])

array_selected = np.array(target)
print(array_selected)


# In[60]:


print(array1d[array1 > 5])


# ### 행렬의 정렬 – sort( )와 argsort( )
# 
# * 행렬 정렬

# In[61]:


org_array = np.array([ 3, 1, 9, 5]) 
print('원본 행렬:', org_array)

# np.sort( )로 정렬 
sort_array1 = np.sort(org_array)         
print ('np.sort( ) 호출 후 반환된 정렬 행렬:', sort_array1) 
print('np.sort( ) 호출 후 원본 행렬:', org_array)

# ndarray.sort( )로 정렬
sort_array2 = org_array.sort()
org_array.sort()
print('org_array.sort( ) 호출 후 반환된 행렬:', sort_array2)
print('org_array.sort( ) 호출 후 원본 행렬:', org_array)


# In[62]:


sort_array1_desc = np.sort(org_array)[::-1]
print ('내림차순으로 정렬:', sort_array1_desc) 


# In[63]:


array2d = np.array([[8, 12], 
                   [7, 1 ]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2d_axis1)


# * argsort

# In[64]:


org_array = np.array([ 3, 1, 9, 5]) 
print(np.sort(org_array))

sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices)


# In[65]:


org_array = np.array([ 3, 1, 9, 5]) 
print(np.sort(org_array)[::-1])

sort_indices_desc = np.argsort(org_array)[::-1]
print('행렬 내림차순 정렬 시 원본 행렬의 인덱스:', sort_indices_desc)


# key-value 형태의 데이터를 John=78, Mike=95, Sarah=84, Kate=98, Samuel=88을 ndarray로 만들고
# argsort()를 이용하여 key값을 정렬

# In[66]:


name_array=np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array=np.array([78, 95, 84, 98, 88])

# score_array의 정렬된 값에 해당하는 원본 행렬 위치 인덱스 반환하고 이를 이용하여 name_array에서 name값 추출.  
sort_indices = np.argsort(score_array)
print("sort indices:", sort_indices)

name_array_sort = name_array[sort_indices]

score_array_sort = score_array[sort_indices]
print(name_array_sort)
print(score_array_sort)


# ### 선형대수 연산 – 행렬 내적과 전치 행렬 구하기
# 
# * 행렬 내적

# In[67]:


A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

dot_product = np.dot(A, B)
print('행렬 내적 결과:\n', dot_product)


# * 전치 행렬

# In[68]:


A = np.array([[1, 2],
              [3, 4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬:\n', transpose_mat)


# In[ ]:




