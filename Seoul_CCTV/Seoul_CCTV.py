#*************************************
# 서울시 각 구별 CCTV 수를 파악하고, 인구대비 CCTV 
# 파악해서 순위를 비교한후 관련 그
# 
#*************************************
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import pandas as pd

# shift + alt + F5, 한줄씩 실행: ctrl + enter
# CCTV 현황데이터 출처: http://data.seoul.go.kr/dataList/datasetView.do?infId=OA-2734&srvType=S&serviceKind=1
seoul_cctv = pd.read_csv('C:\\Users\\ezen\\source\\repos\\seoul_cctv\\seoul_cctv.csv',encoding='UTF-8')
seoul_cctv.head()

seoul_cctv_idx = seoul_cctv.columns
seoul_cctv_idx
"""
Index(['기관명','소계','2013년도 이전','2014년','2015년','2016년'], dtype='object')
"""
seoul_cctv.rename(columns = {seoul_cctv.columns[0]:'구별'}, inplace = True)
# inplace = True는 실제 변수의 내용
seoul_cctv.head()

# 서울시 인구 출처: https://data.seoul.go.kr/dataList/datasetView.do?serviceKind=2&infId=419&srvType=S&stcSrl=419

# xls 파일은 xlrd라는 라이브러리를 import해야 함.
import xlrd
seoul_pop = pd.read_excel("C:\\Users\\ezen\source\\repos\\Seoul_CCTV\\seoul_pop.xls",encoding='UTF-8',header=2, usecols='B,D,G,J,N')
seoul_pop.head()
seoul_pop.rename(columns={seoul_pop.columns[0]:'구별',
                          seoul_pop.columns[1]:'인구수',
                          seoul_pop.columns[2]:'한국인',
                          seoul_pop.columns[3]:'외국인',
                          seoul_pop.columns[4]:'고령자'
                          }, inplace=True)
seoul_pop.head()
import numpy as np
seoul_cctv.sort_values(by='소계', ascending = True).head()
"""
CCTV의 전체 갯수가 가장 적은 구는 도봉구, 강북구, 광진구, 강서구, 중랑구
"""
seoul_cctv.sort_values(by='소계', ascending=False).head()
"""
CCTV의 전체 갯수가 가장 많은 구는 강남구, 양천구, 서초구, 관안구, 은평구
"""
# 서울 인구표의 0번째 합계는 필요없는 값 제거하기
seoul_pop.drop([0],inplace=True)
seoul_pop.head()
# 전체 구의 목록을 출력
seoul_pop['구별'].unique()
"""
array(['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구',
       '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구',
       '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'], dtype=object)
"""
# NaN 값을 제거하기
seoul_pop[seoul_pop['구별'].isnull()]
#**********************
# 외국인 비율과 고령자 비율 계산
#**********************
seoul_pop['외국인비율'] = seoul_pop['외국인']/seoul_pop['인구수']*100
seoul_pop['고령자비율'] = seoul_pop['고령자']/seoul_pop['인구수']*100
seoul_pop.head()
#**********************
# CCTV 데이터와 인구 현황 데이터 합치고 분석하기
#**********************
data_result = pd.merge(seoul_cctv,seoul_pop,on='구별')
data_result.head()
# 그래프로 그리기 위해서는 구이름을 인덱스로 설정

data_result.set_index('구별',inplace=True)
data_result.head()

np.corrcoef(data_result['고령자비율'],data_result['소계'])
"""
array([[ 1.        , -0.27533083],
       [-0.27533083,  1.        ]])
"""
np.corrcoef(data_result['외국인비율'],data_result['소계'])
"""
array([[ 1.        , -0.04796912],
       [-0.04796912,  1.        ]])
"""
np.corrcoef(data_result['인구수'],data_result['소계'])
"""
array([[1.       , 0.2242953],
       [0.2242953, 1.       ]])
"""
# CCTV와 고령자비율은 약한 음의 상관관계 
# 외국인비율은 상관관계 없음
# 인구수와는 약한 양의 상관관계를 가진다.

#한글깨짐 방지
#path = 'C:\\Windows\\Fonts\\malgun.ttf'
path = 'C:\\Windows\\Fonts\\H2GTRM.TTF'
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font',family=font_name)
data_result['CCTV비율'] = data_result['소계']/data_result['인구수']*100
data_result['CCTV비율'].sort_values().plot(kind='barh', grid=True,figsize=(10,10))
# barh 수평 바 차트
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'],data_result['소계'],s=50)
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

# CCTV와 인구수는 양의 상관관계이므로 직선을 그릴수 있다.

fp1 = np.polyfit(data_result['인구수'],data_result['소계'],1)
#polyfit은 직선구하기 명령
fp1
# array

f1 = np.poly1d(fp1) # y축 데이터
# poly + 숫자1 + d
fx = np.linspace(100000, 700000, 100) # x축 데이터

plt.figure(figsize = (10,10))
plt.scatter(data_result['인구수'],data_result['소계'],s=50)
plt.plot(fx,f1(fx),ls = 'dashed', lw = 3, color = 'g')
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()
# 이 데이터에서 직선이 전체 데이터의 대표값 역할을 한다면
# 인구수가 400000 일때 CCTV는 1500 대 정도여야 한다는 결론을
# 내리게 된다.
# 오차를 계산할 수 있는 코드를 만들고, 오차가 큰 순으로 데이터를 정렬

fp1 = np.polyfit(data_result['인구수'],data_result['소계'],1)

f1 = np.poly1d(fp1)
# 숫자 1 주의
fx = np.linspace(100000, 700000, 100)

data_result['오차'] = np.abs(data_result['소계'] - f1(data_result['인구수']))
df_sort = data_result.sort_values(by='오차',ascending=False)
df_sort.head()

plt.figure(figsize = (14,10))
plt.scatter(data_result['인구수'],data_result['소계'],c=data_result['오차'],s=50)
plt.plot(fx,f1(fx),ls = 'dashed', lw = 3, color = 'g')

for n in range(10):
    plt.text(df_sort['인구수'][n]*1.02,df_sort['소계'][n]*0.98,df_sort.index[n],fontsize=15)

plt.xlabel('인구수')
plt.ylabel('인구당비율')

plt.colorbar()
plt.grid()
plt.show()

data_result.to_csv('C:\\Users\\ezen\\source\\repos\\Seoul_CCTV\\data-result.csv')

