"""
타이타닉호의 침몰 당시의 승객 명단 데이터를 통해 생존자의 이름, 성별, 나이, 티켓요금,
생사여부의 정보를 획득합니다. 이를 분석하여 각각의 데이터들간의 연관성을 분석하여
생존에 영향을 미치는 요소를 찾아내는 것.
데이터는 train.csv(훈련데이터)와 test.csv(목적데이터) 두개가 제공됩니다.
목적데이터는 훈련데이터에서 Survived 즉, 생존여부에 대한 정보가 빠져있습니다.

즉, 훈련데이터에 있는 정보를 통해서 적합한 분석 model을 구성한 뒤
이를 목적데이터에 반영하여 생존여부를 추측하는 과정을 수행하고자 합니다.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('C:\\Users\\ezen\\source\\repos\\Seoul_CCTV\\train.csv')
test = pd.read_csv('C:\\Users\\ezen\\source\\repos\\Seoul_CCTV\\test.csv')
train.head()
test.head()
train.columns
"""
'PassengerId', - 승객번호
Survived - 생존여부 (0 사망, 1 생존)
'Pclass', -  승선권 클래스 (1 -1등석, 2- 2등석, 3 - 3등석)
'Name', -승객이름
'Sex', - 생객 성별
'Age', -  승객 나이
'SibSp', - 동반한 형제, 자매, 배우자 수
'Parch',   - 동반한 부모, 자식 수
'Ticket', - 티켓의 고유 넘버
'Fare', - 티켓의 요금
'Cabin', - 객실 번호
'Embarked' - 승선한 항구명
        (C 캠브릿지, Q 퀸스타운, S 사우스햄프턴)
"""
f, ax = plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train, ax=ax[1])
ax[1].set_title('Survived')
plt.show()
"""
탑승객의 60% 이상이 사망했음 (0 사망, 1 생존)
"""
f, ax = plt.subplots(1,2,figsize=(18,8))
train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('남성 생존자')
ax[1].set_title('여성 생존자')

plt.show()
"""
남자의 사망율은 80%, 여자의 사망율은 25%
"""
# 성별과 객실 클래스와의 관계 시트 생성하려고 할대, crosstab 사용함
df_1 = [train['Sex'],train['Survived']]
df_2 = train['Pclass']
pd.crosstab(df_1, df_2, margins = True)

"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""
#1등객실 여성의 생존율은 91/94 = 97%
#3등개실 여성의 생존율은 50%
#1등객실 남성의 생존율은 37%
#3등객실 남성의 생존율은 13%

# 배들 탄 항구와의 연관성 추출
f, ax = plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=train,ax = ax[0,0])
ax[0,0].set_title('승선한 인원')
sns.countplot('Embarked',hue='Sex',data=train,ax=ax[0,1])
ax[0,1].set_title('승선한 성별')
sns.countplot('Embarked',hue='Survived',data=train, ax=ax[1,0])
ax[1,0].set_title('승선한 항구 vs 생존자')
sns.countplot('Embarked',hue='Pclass',data=train, ax=ax[1,1])
ax[1,1].set_title('승선한 항구 vs 객실등급')
plt.show()
"""
절반 이상의 승객이 사우스햄프턴에서 배를 탔으며, 여기에 탑승한 승객의 70% 가량이
남성이었습니다. 남성의 사망율이 여성보다 훨씬 높았으므로 사우스햄프턴에서

"""

# *************
# 
# *************


"""

"""

# 모델을 만들
train.info

train.isnull().sum()
"""
Age         177의 결측치
Cabin       687의 결측치
Embarked    2의 결측치
나이는 생존율에 민감하므로 임의의 데이터로 채운다.
객실번호는 임의의 데이터로 산정하기 어렵고, 결측치가 너무 많아
제거하기로 한다.
승선한 항구 2개의 결측치는 수가 적으므로 임의의 값으로 대체한다.
"""

sns.set()

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.show()

bar_chart('Sex')
bar_chart('Pclass') #사망자는 3등석, 생존자는 1등석
bar_chart('SibSp') #동반한 형제자매, 배우자수
bar_chart('Parch') #동반한 부모, 자식수
bar_chart('Embarked') #승선한 항구
# S, Q에 탑승한 사람이 더 많이 사망했고, C는 덜 사망했다.

"""
Feature Engineering은 머신러닝 알고리즘을 작동하기 위해 
데이터에 특징을 만드는 과정.
모델의 성능을 높이기 위해 모델에 입력할 데이터를 만들기 위해
주어진 초기 데이터로부터 특정을 가공하고
생성하는 전체 과정을 의미합니다.
"""

"""
위 정보에서 얻을 수 있는 사실은 아래와 같습니다.
1. Age의 약 20프의 데이터가 Null로 되어있다.
2. Cabin의 대부분 값은 Null이다.
3. Name, Sex, Ticket, Cabin, Embarked는 숫자가 아닌 문자 값이다.
   - 연관성 없는 데이터는 삭제하거나 숫자로 바꿀 예정입니다.
     (머신러닝은 숫자를 인식하기 때문입니다.)
그리고 이를 바탕으로 이렇게 데이터를 가공해 보겠습니다.
1. Cabin과 Ticket 두 값은 삭제한다.(값이 비어있고 연관성이 없다는 판단하에)
2. Embarked, Name, Sex 값은 숫자로 변경할 것 입니다.
3. Age의 Null 데이터를 채워 넣을 것입니다.
4. Age의 값의 범위를 줄일 것입니다.(큰 범위는 머신러닝 분석시 좋지 않습니다.)
5. Fare의 값도 범위를 줄일 것입니다.
"""

# Cabin, Ticket 값 삭제
train = train.drop(['Cabin'],axis=1)
test = test.drop(['Cabin'],axis=1)
train = train.drop(['Ticket'],axis=1)
test = test.drop(['Ticket'],axis=1)
train.head()
test.head()

# Embarked 값 가공
s_city = train[train['Embarked']=='S'].shape[0]
print("S : ",s_city) # S : 646
s_city = train[train['Embarked']=='C'].shape[0]
print("C : ",s_city) # C : 168
s_city = train[train['Embarked']=='Q'].shape[0]
print("Q : ",s_city) # Q : 77

"""
대부분의 값이 S이므로 결측값 2개도 S로 채우는 것으로 결정
"""
train = train.fillna({'Embarked':'S'})
"""
S- 1, C-2, Q-3으로 변경, 머신러닝은 숫자만 인식함.
"""
city_mapping = {"S":1, "C":2,"Q":3}
train['Embarked'] = train['Embarked'].map(city_mapping)
test['Embarked'] = test['Embarked'].map(city_mapping)

train.head()
test.head()

# Name 값 가공하기

combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')
    # ([A-Za-z]+)\.은 정규식. []은 글자단위인데 알파벳만 허용함. +는 한 글자 이상. \.은 
    # 글자 뒤에 반드시점(.)이 옴
pd.crosstab(train['Title'],train['Sex'])

"""
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
"""

"""
Mr, Mrs, Miss, Royal, Rare, Master 6개로 줄여봄. 이를 바탕으로 생존율의 평균을 살펴봄.
"""
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Sir','Lady'],'Royal')
    dataset['Title'] = dataset['Title'].replace(['Mile','Ms'],'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()

"""
    Title  Survived
0  Master  0.575000
1    Miss  0.699454
2    Mlle  1.000000
3      Mr  0.156673
4     Mrs  0.793651
5    Rare  0.250000
6   Royal  1.000000
이 데이터를 바탕으로 1부터 6까지 숫자로 매핑함.
"""

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train.head()




# Name과 PassengerId 삭제
train = train.drop(['Name','PassengerId'],axis = 1)
test = test.drop(['Name','PassengerId'],axis = 1)
train = train.drop(['Age','Fare'],axis = 1)
test = test.drop(['Age','Fare'],axis = 1)
combine = [train, test]
train.head()

sex_mapping = {'male':0, 'female':1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train.head()

train_data = train.drop('Survived', axis = df_1)
target = train['Survived']
train_data.shape, target.shape

# ((891,8),(891,))

train.info

# 현재 train의 정보가 최종 모델의 모습
# NaN이 없음, 전부 숫자값으로 매핑된 상황










