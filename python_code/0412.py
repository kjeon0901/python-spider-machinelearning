import csv
import matplotlib.pyplot as plt 	#matplotlib패키지 안의 pyplot모듈(그래프 그리기 위해서 필요)을 불러와 여기서 plt라 부르겠다 

f=open("C:/jeon/ingu.csv")
main_data=csv.reader(f)
temp=[]
for row in main_data:
    temp.append(row)
f.close()
main_pt=temp[1:]


#청담동의 총 인구 수를 int로 출력
for idx, row in enumerate(main_pt):
    if '청담' in row[0]:
        print("청담동 총 인구 수 :", int(row[1].replace(",", '')))


#총 인구 수가 가장 많은 동을 찾아서 출력
maxofdong=int(main_pt[2][1].replace(",", ''))
maxidx=2
for idx, row in enumerate(main_pt):
    #(나오기 직전까지 잘라서 마지막에 공백 있으면 continue, 아닐 때만 생각
    if row[0][:row[0].index('(')][-1]==" ":
        continue
    newrow1 =int(row[1].replace(",", ''))
    if newrow1>maxofdong:
        maxofdong=newrow1
        maxidx=idx
print("총 인구 수가 가장 많은 동 :", main_pt[maxidx][0], maxofdong, "명")


#그래프 그리기 : spider의 Plots 탭에서 확인 가능 
A=[1,5,8,8,4,3,2]
plt.plot(A) 		#plot()의 인수로는 수치적으로 판단 가능한 타입만 


#신중동에서 0~100세이상 까지의 그래프 그리기
shinjoong_bothsex =[]   	#슬라이싱 바로 하니까 int형으로 각각 넣을 수가 없어서 이렇게 바꿈
for x in range(3, 104):
    shinjoong_bothsex.append(int(main_pt[maxidx][x].replace(",", '')))
plt.plot(shinjoong_bothsex)


#xx동 입력받고 있으면 인덱스 구하기
dong=input("동 이름을 입력하세요:")         
for idx, row in enumerate(main_pt):
    if row and row[0]:
        if ' '+dong+'(' in row[0]:
            break
if idx==3845 and '예래동'!= dong:
    print("해당하는 동이 없습니다.")
else:       #xx동에서 0~100세이상 까지의 그래프 그리기
    dong_bothsex=[]
    for x in range(3, 104):
        dong_bothsex.append(int(main_pt[idx][x].replace(',', '')))
    plt.plot(dong_bothsex)
    
    
#신중동과 가장 유사한 그래프 모양을 가진 동 찾기
'''
ex
           0세      1세      2세      3세
신중동     100      200      150      180
청담동      80      150      170      100
효자동    1000     2000     1500     1800

차이는 청담동이 덜 나지만, 비율로 따져야 함. 효자동이 정답!
abs(신중동 0세/총인구 - xx동 0세/총인구) + abs(신중동 1세/총인구 - xx동 1세/총인구) + ...
'''
ratio_min_different=[main_pt[0][0], 100] #비율차이 가장 적은 동 이름, 그 동과의 비율차이 담을 변수 초기화
shinjoong_ratio=list(map(lambda x:x/sum(shinjoong_bothsex), shinjoong_bothsex))
for idx, row in enumerate(main_pt):
    if row[0][:row[0].index('(')][-1]==" ":
        continue
    if row[0]=='경기도 부천시 신중동(4119074200)':
        continue
    
    each_bothsex=[]
    for x in range(3, 104):
        each_bothsex.append(int(row[x].replace(",",'')))
    if row and row[0] and sum(each_bothsex)>0:
        each_ratio=list(map(lambda x:x/sum(each_bothsex), each_bothsex))
    
    sum_all_abs=0
    for n in range(len(shinjoong_ratio)):
        sum_all_abs += abs(shinjoong_ratio[n]-each_ratio[n])
    if sum_all_abs < ratio_min_different[1]:
        ratio_min_different[0]=row[0]
        ratio_min_different[1]=sum_all_abs
print(ratio_min_different)