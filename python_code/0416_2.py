
'''
가장 많은 오류
-No such file or directory: '없는파일이름'
-division by zero
-list index out of range
cf. None 넣어주는 건 오류 x!!
	def add()의 리턴값이 없을 때 c=add()는 오류 x

예외처리 1. try, except문
try: except:			//모든 예외처리
try: except Exception as e:		//모든 예외처리 - 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
try: except FileNotFoundError as e: 	//No such file or directory만 예외처리. 나머지는 오류남
try: except ZeroDivisionErrow as e: 	//division by zero만 예외처리. 나머지는 오류남
try: except IndesErrow as e: 	//list index out of range만 예외처리. 나머지는 오류남
        cf. 	except문 여러개 쓰면 여러개 예외처리 할 수 있지만, 
	앞에서 오류가 발생했으면 이미 멈춰서(except문으로 이미 빠져버린다) 
	다음 코드는 실행되지도 않고 오류도 발생하지 않는다.

예외처리 2. try, finally문
형태는 똑같지만 try문 수행 도중 예외가 발생했든 아니든(try문 수행하든 except문 수행하든) finally문 항상 수행

raise 키워드로 오류 일부러 발생시키기
상속해주는 부모 클래스의 특정 메소드가 문제가 있어서 상속받은 자식 클래스에서 무조건 메소드 오버라이딩을 해줬으면 하는 상황에 쓰임
raise 에러 이름 → 해당 에러 발생
	ex_ raise NotImplementedError → 꼭 작성해야 하는 부분이 구현되지 않았을 때 발생
				    → 부모 클래스 메소드에 써둠. 자식 클래스에서 해당 메소드 구현하지 않을 경우 호출


'''