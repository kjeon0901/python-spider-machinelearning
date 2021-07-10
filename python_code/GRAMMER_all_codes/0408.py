import csv                  #csv라는 패키지(csv 파일_:통계데이터 많이 담음_을 처리할 수 있게 해줌) 가져옴

f=open('C:/jeon/gihu.csv') #이 경로의 파일을 열고 불러와 객체로 f에 담기 (더블클릭 따닥-!)
main_data = csv.reader(f)   #내장함수/패키지에구현된함수/모듈안의함수 모두 .을 통해 불러옴
                            #f를 육안으로 확인하기 위해 csv.reader()함수의 인수로 넣어 리턴값을 main_data에 넣음
print(main_data)            #<_csv.reader object at 0x000002CD479F6A00> 출력됨(매번바뀜)
                            #위의 16자리 16진수 자리의 주소(16==2^4이므로 16진수 1자리==2진수 4자리. 즉 16진수 16자리==2진수 64자리==64bit : 메모리 한줄에 64bit 크기의 데이터를 넣음)에 저장되어 있다는 뜻

#필요한 데이터만 main_pt에 모으기
temp=[]                     #main_data가 reader타입이기 때문에, temp라는 리스트에 리스트 타입으로 바꿔 담아줄거임 
for row in main_data:       #순차적인 요소가 있는 main_data에서 한줄씩 row에 불러와서
    temp.append(row)        #data라는 리스트에 row를 순차적으로 추가
f.close()                   #파일 닫기

main_pt=temp[8:]           #temp의 핵심 데이터(12번째 row부터 끝 row까지)만 담기

'''
참고 1. 
리스트 슬라이싱은 자동으로 복사를 해와서 가져오는 것이기 때문에 main_pt=temp[12:]한다고 해서 a=b에서 아예 동일한 주소를 참조하게 되는 것과는 다름!! main_pt[0] = 1 해보면 main_pt에서만 바뀐다 

참고 2.
temp=[]
for row in main_data[12:]: #참고로 얘는 이렇게 슬라이싱이 안 된다. 오류남... 
    temp.append(row)
'''

#4번째 index만 비교해서 최고기온의 최댓값과 날짜 구하기
max=float(main_pt[0][4])
for idx, row in enumerate(main_pt):
    try:
        if row:         #오류 원인 2 해결 ========> row의 타입은 list. list가 True(==요소가 하나라도 있다)인 경우만 거르기 
            if row[4]:  #오류 원인 1 해결 ========> row[4]의 타입은 string. string이 True(==''이 아니다)인 경우만 거르기
                if float(row[4])>max:
                    max=float(row[4])
                    max_index=idx   #최댓값을 max에 업데이트하는 바로 그 순간의 for loop의 index. 
    except Exception as e:
        print(idx, e)   #어디서 오류 발생했는가 => 이 부분 출력해보면 오류 원인 알 수 있음 ↓↓↓
#print(main_pt[39758])  ========> 39758 could not convert string to float: ''    오류 원인 1 => ['2017-10-12', '108', '11.4', '8.8', '']. 4번째인덱스 비어있구나~
#print(main_pt[41033])  ========> 41033 list index out of range                  오류 원인 2 => []. 아예 리스트가 비어 있었구나~
print(main_pt[max_index][0], max)

'''
내가 했던 방법 1.
max=float(main_pt[0][4])
for x in range(1, len(main_pt)):
    if len(main_pt[x])<4:   #오류처리 인덱스 4까지 없을 때
        continue
    if main_pt[x][4]=='':   #오류처리 인덱스 4에 아무 값도 안 들어있을 때
        continue
    if float(main_pt[x][4])>max:
        max=float(main_pt[x][4])
print(max)

내가 했던 방법 2.
max=float(main_pt[0][4])
for row in main_pt:
    if len(row)<4 or row[4]=='':    #오류처리 인덱스가 4까지 없거나 빈 string일 때
        continue
    if float(row[4])>max:
        max=float(row[4])
print(max)

참고 1.
enumerate : for문이 돌면서 loop 횟수에 인덱스를 0, 1, 2, ... 부여해 idx에 넣는다. row에는 원래 for문처럼 temp의 요소 하나씩 가져온다. 
enumerate는 많이 쓰이니까 그냥 습관적으로 for문에 붙여주면 됨.

참고 2.
for idx, row in enumerate(temp):
    try:                    #try-except: try문에 묶인 코드 상에서 만약 에러가 발생했다면, 일단 실행은 stop하지 말고 try문 스킵한 다음에 except문을 대신 실행해줘!!!
        if row[4]:
            float(row[4])
    except Exception as e:  #에러가 발생하지 않는다면 실행되지 않는다 
        print(idx, e)
        print(row)
'''