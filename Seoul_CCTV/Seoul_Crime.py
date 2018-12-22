"""
서울시 범죄 현황 분석
"""
import numpy as np
import pandas as pd
import googlemaps
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns




#crime_anal_police = pd.read_csv('C:\\Users\\ezen\source\\repos\Seoul_CCTV\\seoul_crime.csv',thousands=',',encoding = 'euc-kr')
crime_anal_police = pd.read_csv('C:\\Users\\ezen\source\\repos\Seoul_CCTV\\crime_in_Seoul.csv',thousands=',',encoding = 'euc-kr')

crime_anal_police.head()

gmaps_key = "AIzaSyBsaB4ap6Ti2JLjGjn6Vx3n1IquK5zmNnQ"
gmaps = googlemaps.Client(key = gmaps_key)

gmaps.geocode('서울중부경찰서',language='ko')

"""

"""

# formatted_address
# lng 위도
# lat 경도

station_name = []
for name in crime_anal_police['관서명']:
    station_name.append('서울'+str(name[:-1])+'경찰서')
station_name
station_address = []
station_lat = []
station_lng = []
for name in station_name:
    tmp = gmaps.geocode(name, language='ko')
    station_address.append(tmp[0].get('formatted_address'))
    tmp_loc = tmp[0].get('geometry')
    station_lat.append(tmp_loc['location']['lat'])
    station_lng.append(tmp_loc['location']['lng'])
    print(name + '----->'+tmp[0].get('formatted_address'))
station_lat
station_lng
gu_name = []
# 2차 코딩


for name in station_address:
    tmp = name.split()

    tmp_gu = [gu for gu in tmp if gu[-1]=='구'][0]
    print('****'+tmp_gu)
    gu_name.append(tmp_gu)
gu_name

type(crime_anal_police) #pandas.core.frame.DataFrame
len(crime_anal_police['관서명']) #31
type(gu_name) #'list'
len(gu_name) #31
crime_anal_police.head()

# 금천경찰서는 관악구 위치에 있어서 금천서는 예외 처리

crime_anal_police[crime-anal_police['관서명']=='금천서']

crime_anal_police.loc[crime_anal_police['관서명']=='금천서',['구별']] = '금천서'
# 금천서를 찾아서 
crime_anal_police[crime_anal_police['관서명']=='금천서']
# 관악구로 되어 있는 것을 금천구로 바꿔라

crime_anal_police

# 중간에 에러가 나서 계속 데이터를 제작하는 것을 방지하기 위ㅐ
# 2번 데이터로 저장

crime_anal_police.to_csv('C:\\Users\\ezen\\source\\repos\\Seoul_CCTV\\crime_anal_police2.csv')
crime_anal_police2 = pd.read_csv('C:\\Users\\ezen\\source\\repos\\Seoul_CCTV\\crime_anal_police2.csv')
crime_anal_police2

# 관서별로 되어 있는 것을 구별로 바꾸는 작업

crime_anal_police3 = pd.pivot_table(crimecrime_anal_police2, index='구별',aggfunc=np.sum)

crime_anal_police3.head()

police = crime_anal_police3

police['강간검거율'] = police['강간 검거']/police['강간 발생']*100
police['강도검거율'] = police['강도 검거']/police['강도 발생']*100
police['절도검거율'] = police['절도 검거']/police['절도 발생']*100
police['살인검거율'] = police['살인 검거']/police['살인 발생']*100
police['폭력검거율'] = police['폭력 검거']/police['폭력 발생']*100

del police['강간 검거']
del police['강도 검거']
del police['절도 검거']
del police['살인 검거']
del police['폭력 검거']

con_list = ['강간검거율','강도검거율','절도검거율','살인검거율','폭력검거율']
con_list
for column in con_list:
    police.loc[police[i] > 100, i] = 100 
~police.head()

    #검거율이 100이 넘는 값이 보임. 1년 이상의 기간이 포함된 데이터 오류
    #비율이 100을 넘을 수 없으니 100 오버는 그냥 100으로 처리 

police.rename(columns = {'강간 발생':'강간',
                         '강도 발생':'강도',
                         '절도 발생':'절도',
                         '살인 발생':'살인',
                         '폭력 발생':'폭력'
                         }, inplace = True)

# 숫자값으로 모델링화

col = ['강간','강도','절도','살인','폭력']

x = police[col].values
min_max_scalar = preprocessing.MinMaxScaler(0)
"""
스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에
선형 변환을 적용하여 전체 자료의 분포를
평균 0, 분산 1이 되도록 만드는 과정이다.
"""

x_scaled = min_max_scalar.fit_transform(x.astype(float))
# min_max_scale(X): 최대/최소값이 각각 1, 0이 되도록 스케일링
police_norm = pd.DataFrame(x_scaled, columns=col, index = police.index)
# 각 컬럼별로 정규화 하기

col2 = ['강간검거율','강도검거율','절도검거율','살인검거율','폭력검거율']
police_norm[col2] = police[col2]
police_norm.head()

"""

"""
# 발생건수를 정규화 시켰다.

data_result = pd.read_csv('C:\\Users\\ezen\\source\\repos\\Seoul_CCTV\\data_result.csv')
data_result.head()


police_norm['범죄'] = np.sum(police_police_norm[col],axis = 1)
police_norm.head()

police_norm['검거'] = np.sum(police_police_norm[col2],axis = 1)
police_norm.head()






