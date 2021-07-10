'''

★
____________________________[파이썬 내장함수]____________________________
abs(x) → x의 절댓값
pow(x, y) → x^y 
round(x, y) → x를 소숫점 y째 자리까지 반올림 (y 없어도 됨) 
sum(s) → s의 모든 요소의 합
all(x)♣ → x의 모든 '요소'가 참일 때 True, 하나라도 거짓일 때 False
any(x)♣ → x의 하나의 '요소'라도 참일 때 True, 모두 거짓일 때 False
chr(아스키코드값), ord(문자) → 아스키코드에 해당하는 문자 리턴, 문자에 해당하는 아스키코드값 리턴
int(x), hex(x), oct(x) → 정수 x를 10진수, 16진수, 8진수로 변환해서 리턴
	cf___int('0xea', 16)  #지금 '0xea'는 16진수로 표기되어 있는데, 얘를 10진수로 바꿔주세요!
id(x) → 객체 x의 주소값
len(s) → s의 길이, 즉 s 안의 요소의 전체 개수
list(s) → s를 리스트로 만들어 리턴
dir(x) → x의 자료형이 사용할 수 있는 모든 내장함수 리턴
enumerate → for loop 돌린 횟수에 인덱스를 부여. 습관적으로 사용하자!
filter(함수명a, x)♣ → x의 요소 중 함수a 돌렸을 때 리턴값이 참인 것만 묶어서(걸러내서) 리턴
		→ x는 iterate(여러개의 요소 가짐)여야 함!
	>>>def positive(x):
	          return x>0	//양수면 True, 그게 아니면 False
	      print( list( filter(positive, [1,-3,2,0,-5,6])))
	>>>print( filter( lambda x : x>0, [1,-3,2,0,-5,6]))	    //lambda a, b : a+b → input값 a, b, output값 a+b
						    //이러면 함수a 말하면서 x도 써주니까 lambda만 넣어주면 됨
	------------------------------------------------------
	>>> [1, 2, 6]
	>>> [1, 2, 6]
map(함수명a, x)♣ → x의 모든 요소에 a효과를 넣어줌
		→ x는 iterate(여러개의 요소 가짐)여야 함!
	>>>def two_times(x):
	          return x*2
	      print( list( map(two_times, [1,2,3,4]))
	>>>print( map( lambda a : a*2, [1,2,3,4]))
            -------------------------------------------------------
	>>>[2, 4, 6, 8]
	>>>[2, 4, 6, 8]
sorted(s) → s의 요소들을 정렬한 뒤 리스트로 리턴	//리스트.sort()는 리턴해주지는 않는다
zip(동일한 len의 여러 자료)♣ → 같은 인덱스의 데이터끼리 묶어줌
★

모듈 : 파이썬 파일(.py)
여러 모듈(import해오는 것들)이 합쳐지면 : 패키지
여러 패키지가 합쳐지면 : 라이브러리

대화형 인터프리터 : Anaconda prompt 켜서 python 입력하면 >>>형태로 실행됨
if __name__ == "__main__" 이거 여기서 사용하는 것

__init__.py
해당 디렉터리가 패키지의 일부임을 알려준다. 
python3.3버전부터는 굳이 없어도 알아서 "이 디렉터리가 패키지를 위한 것이구나~" 인식해서 오류 x
그래도 python3.3 이하의 하위 버전에서 도는 애플리케이션까지 호환되면 좋으니까 그냥 __init__.py 쓰자!

A디렉터리의 a.py모듈이 B디렉터리의 b.py모듈을 사용하고 싶다면?
a.py 모듈 안에서 from B import b.py 해주고 아래에서 b.py모듈 안의 함수를 사용하면 됨


'''