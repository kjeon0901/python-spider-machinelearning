import csv

f=open('C:/jeon/jihacheol.csv')
main_data=csv.reader(f)
temp=[]
for row in main_data:
    temp.append(row)
main_pt=temp[2:]
f.close()


#출근시간 7~9시에 가장 많은 하차 인원(11idx, 13idx만 확인)이 카운팅되는 역 찾기
maxgetoff_7to9=int(main_pt[0][11].replace(",", ''))+int(main_pt[0][13].replace(",", ''))
maxidx_7to9=0
for idx, row in enumerate(main_pt):
    if row and row[3] and row[11] and row[13]:
        newgetoff=int(row[11].replace(",", ''))+int(row[13].replace(",", ''))
        if newgetoff>maxgetoff_7to9:
            maxgetoff_7to9=newgetoff
            maxidx_7to9=idx
        
print(main_pt[maxidx_7to9][3], maxgetoff_7to9)
print("7~8시 :",main_pt[maxidx_7to9][11])
print("8~9시 :",main_pt[maxidx_7to9][13])


#모든 역에 대해 [역이름, 승차총합(모든시간대), 하차총합]을 요소르 담는 리스트 만들기
def getsum(n):			#함수 쓰면 리스트 다시 안 비워줘도 되니까 better
    sum_all=0
    for idx2, column in enumerate(row[4:52]):
        if n:
            if not idx2 %2:
                sum_all+=int(column.replace(',',''))
        else:
            if idx2 %2:
                sum_all+=int(column.replace(',',''))
    return sum_all
sum_all_onoff=[]
for idx, row in enumerate(main_pt):
    each_all_onoff=[]
    each_all_onoff.append(row[3])    	#역이름 
    each_all_onoff.append(getsum(1)) 	#승차총합
    each_all_onoff.append(getsum(0)) 	#하차총합
    sum_all_onoff.append(each_all_onoff)


#시간대별 가장 많은 승차인원, 가장 많은 하차인원 리스트로 만들기
#[시간, 역이름, 승차인원, 역이름, 하차인원]
def get_max(n): #(n인덱스가 최대인 역이름, 그 역의 n인덱스값) 튜플로 리턴
    max_idx=0
    max_men=int(main_pt[0][n].replace(",",''))
    for idx, row in enumerate(main_pt):
            if int(row[n].replace(",",''))>max_men:
                max_idx=idx
                max_men=int(row[n].replace(",",''))
    return main_pt[max_idx][3], max_men
def get_time(n): #시간 구하기
    time=n+4
    if time>=24:
        time-=24
    return str(time)+"시"
final=[]
for x in range(24):
    temp=[]    
    max_on=get_max(4+2*x)
    max_off=get_max(5+2*x)
    temp.append(get_time(x))
    temp.append(max_on[0])
    temp.append(max_on[1])
    temp.append(max_off[0])
    temp.append(max_off[1])
    final.append(temp)

'''
#다른방법_1
def get_time(n): #시간 구하기
    time=n+4
    if time>=24:
        time-=24
    return str(time)+"시"
pol=[]
for _ in range(48):
    pol.append([0,0])
for idx, row in enumerate(main_pt):
    for idx2, row2 in enumerate(row[4:-1]):
        if int(row2.replace(",",''))>pol[idx2][1]:
            pol[idx2][0]=row[3]
            pol[idx2][1]=int(row2.replace(",",''))
final1=[]
for x in range(24):
    temp1=[]
    temp1.append(get_time(x))
    temp1.append(pol[x*2][0])
    temp1.append(pol[x*2][1])
    temp1.append(pol[x*2+1][0])
    temp1.append(pol[x*2+1][1])
    final1.append(temp1)


#다른방법_2
MaxMan_counted_list = [0]*48
print(len(MaxMan_counted_list))
station_name_list = [0]*48
temp_list = []
total_list = []

for idx, row in enumerate(main_pt):
    for idx2, row1 in enumerate(row[4:52]):
        if MaxMan_counted_list[idx2] < int(row1.replace(",","")):
            MaxMan_counted_list[idx2] = int(row1.replace(",",""))
            station_name_list[idx2] = str(main_pt[idx][3]) + '_' + str(idx+2)  #그냥 역ID도 같이 출력해봄

for idx, row in enumerate(range(24)):
    temp_list.append(temp[0][idx*2 + 4])
    temp_list.append(station_name_list[idx*2])
    temp_list.append(MaxMan_counted_list[idx*2])
    temp_list.append(station_name_list[idx*2 + 1])
    temp_list.append(MaxMan_counted_list[idx*2 + 1])
    total_list.append(temp_list)
    temp_list = []
'''