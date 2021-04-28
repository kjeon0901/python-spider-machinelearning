import numpy as np
import pandas as pd

array1 = np.array([1,2,3])
print('array1 type:',type(array1))
print('array1 array 형태:',array1.shape) #1차원데이터 (series)
array2 = np.array([[1,2,3],
                  [2,3,4]])
print('array2 type:',type(array2))
print('array2 array 형태:',array2.shape) #2x3 shape를 가진 2차원데이터
array3 = np.array([[1,2,3]])
print('array3 type:',type(array3))
print('array3 array 형태:',array3.shape) #1x3 shape를 가진 2차원데이터 
print('array1: {0}차원, array2: {1}차원, array3: {2}차원'.format(array1.ndim, array2.ndim, array3.ndim))

col_name1=['col1']			# 1개의 컬럼명이 필요함
list1 = [1, 2, 3]
array1 = np.array(list1)
print('array1 shape:', array1.shape )
df_list1 = pd.DataFrame(list1, columns=col_name1)   	#컬럼네임을 col_name1로 지어줌
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
df_array1 = pd.DataFrame(array1, columns=col_name1) 	#컬럼네임을 col_name1로 지어줌
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)

col_name2=['col1', 'col2', 'col3']	# 3개의 컬럼명이 필요함
list2 = [[1, 2, 3],			# 2행x3열 형태의 리스트와 ndarray 생성 한 뒤 이를 DataFrame으로 변환. 
         [11, 12, 13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape )
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array2)

#ndarray는 무조건 동일한 데이터타입 가져야 함.      cf. 리스트 안에는 모든 타입 가능
list1 = [1,2,3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)

list2 = [1, 2, 'test']
array2 = np.array(list2)
print(array2, array2.dtype) #[int, int, str]을 ndarray로 만들면 [str str str]로 타입캐스팅 → ['1' '2' 'test']
                            
list3 = [1, 2, 3.1]
array3 = np.array(list3)
print(array3, array3.dtype) #[int, int, float]을 ndarray로 만들면 [float float float]로 타입캐스팅 → [1. 2. 3.1] 


#0과 1로 초기화, dtype 설정해주지 않으면 default로 float로 들어감
zero_array = np.zeros((3,2),dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)

one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)


#reshape()
array1 = np.arange(10)
print('array1:\n', array1)

array2 = array1.reshape(2,5)
print('array2:\n',array2)

array3 = array1.reshape(5,2)
print('array3:\n',array3)

array4 = array1.reshape(-1,5) #뒷자리가 5로 고정되면 앞자리는 무조건 2만 가능 => 알아서 해주세요~
print('array4 shape:',array4.shape)

array5 = array1.reshape(5,-1) #앞자리가 5로 고정되면 뒷자리는 무조건 2만 가능 => 알아서 해주세요~
print('array5 shape:',array5.shape)


#tolist()
array1 = np.arange(8)
array3d = array1.reshape((2,2,2))
print('array3d:\n',array3d.tolist())

array5 = array3d.reshape(-1,1)
print('array5:\n',array5.tolist())
print('array5 shape:',array5.shape)

array6 = array1.reshape(-1,1)
print('array6:\n',array6.tolist())
print('array6 shape:',array6.shape)
#불린인덱싱
array1d = np.arange(start=1, stop=10)
# [ ] 안에 array1d > 5 Boolean indexing을 적용 
array3 = array1d[array1d > 5]
print('array1d > 5 불린 인덱싱 결과 값 :', array3)


#정렬
org_array = np.array([ 3, 1, 9, 5]) 
print('원본 행렬:', org_array)

sort_array1 = np.sort(org_array)        #sorting된 데이터를 리턴 - 원본 데이터 그대로 냅둠   
print ('np.sort( ) 호출 후 반환된 정렬 행렬:', sort_array1) 
print('np.sort( ) 호출 후 원본 행렬:', org_array)

sort_array2 = org_array.sort()          #원본 데이터 자체를 수정 - 아무것도 리턴 안됨
print('org_array.sort( ) 호출 후 반환된 행렬:', sort_array2)
print('org_array.sort( ) 호출 후 원본 행렬:', org_array)

sort_array1_desc = np.sort(org_array)[::-1]     #:: 는 '거꾸로'
print ('내림차순으로 정렬:', sort_array1_desc) 

array2d = np.array([[8, 12], 
                   [7, 1 ]])
sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)      #8과 7을 비교, 12와 1을 비교
sort_array2d_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2d_axis1)      #8과 12를 비교, 7과 1을 비교

org_array = np.array([ 3, 1, 9, 5]) 
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices)

name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array= np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스:', sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력:', name_array[sort_indices_asc])

#행렬 내적(곱)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

dot_product = np.dot(A, B)
print('행렬 내적 결과:\n', dot_product)



#DataFrame

col_name1=['col1']			# 1개의 컬럼명이 필요함
list1 = [1, 2, 3]
array1 = np.array(list1)
print('array1 shape:', array1.shape )
df_list1 = pd.DataFrame(list1, columns=col_name1)   	#컬럼네임을 col_name1로 지어줌
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
df_array1 = pd.DataFrame(array1, columns=col_name1) 	#컬럼네임을 col_name1로 지어줌
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)

col_name2=['col1', 'col2', 'col3']	# 3개의 컬럼명이 필요함
list2 = [[1, 2, 3],			# 2행x3열 형태의 리스트와 ndarray 생성 한 뒤 이를 DataFrame으로 변환. 
         [11, 12, 13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape )
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array2)



# Key는 컬럼명으로 매핑, Value는 리스트 형(또는 ndarray)
dict = {'col1':[1, 11], 'col2':[2, 22], 'col3':[3, 33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n', df_dict)

# DataFrame을 리스트로 변환
print(df_dict.values)
list3 = df_dict.values.tolist()     #df_dict.values => ndarray로 바뀜. .tolist()로 최종적으로 리스트로 바뀜
                                    #바로 df_dict.tolist() 할 수 없다
print('df_dict.values.tolist() 타입:', type(list3))
print(list3)

# DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('\n df_dict.to_dict() 타입:', type(dict3))
print(dict3)