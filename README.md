# titanic 데이터 분석
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')
#15개의 차원 , 891개의 점
df.shape
``` 

### pclass칼럼의 value_counts()
 ```
df.pclass.value_counts()
```
### 모든 칼럼의 각 항목의 개수로 그래프 그리기
### 연속 데이터는 kind를 'hist'로 해서그리면 보기 좋음
```
for i in df.columns:
    print('%s'%i)
    print(df[i].value_counts())
    #그래프 사이즈
    plt.figure(figsize=(20,10))
    df[i].value_counts().plot(kind='bar')
    plt.show()
```
### 데이터 프레임의 Column,Non-Null Count,Dtype출력     
```py
df.info()
```

\* 참고 \*
1.
```py
import numpy as np
arr=np.array([1,2,3,4,5,6,7,8,9,10])
print(arr.mean())
print(np.median(arr))
```
2.
```py
arr=np.array([1,2,3,4,5,6,7,8,9,1000])
print(arr.mean())
print(np.median(arr))
```
1번에서는 mean=5.5,median =5.5<br>
2번에서는 mean=104.5,median=5.5<br>
median은 이상치나 극단치 판단에 활용하기 좋은 것을 알 수 있다.<br>
만약 mean,median이 비슷하면 이상치가 별로없다는 것을 알 수 있다.<br>

결측치 있는지 확인
```
df.isnull().sum()
```
<img width="89" alt="image" src="https://user-images.githubusercontent.com/88071262/178707875-16ccf61d-3151-401a-83ae-9ae7c2b4b550.png">
age,embarked,deck에 결측치가 있는 것을 확인<br><br>

무한대 있는지 확인=> 거진 NAN값이라 봄
```py
np.isinf(df).sum()
```
<img width="485" alt="image" src="https://user-images.githubusercontent.com/88071262/178707812-7c8ead79-282a-4bad-9428-8f5cc2b34715.png">

embark_town와 embarked의 데이터가 같은것을 의미하기때문에 embark_town컬럼 삭제
```py
#inplace =true 원본이 업데이트됨
df.drop('embark_town',axis=1,inplace=True)
```
그래프를 보면 s의 값이 많음 따라서 embarked의 NAN값을 S로 업데이트 해줄것임<br>
```py
df.embarked.value_counts().plot(kind='bar')
```
NAN값을 S로 업데이트<br>
```py
df.embarked.fillna('S',inplace=True)
```
결측치 업데이트
```py
#1.남성,1등석
male_1_median=df[(df['pclass']==1) & (df['sex']=='male')]['age'].median()
print(male_1_median)
#2.여성,1등석
female_1_median=df[(df['pclass']==1) & (df['sex']=='female')]['age'].median()
print(female_1_median)
#3.남성,2등석
male_2_median=df[(df['pclass']==2) & (df['sex']=='male')]['age'].median()
print(male_2_median)
#4.여성,2등석
female_2_median=df[(df['pclass']==2) & (df['sex']=='female')]['age'].median()
print(female_2_median)
#5.남성,3등석
male_3_median=df[(df['pclass']==3) & (df['sex']=='male')]['age'].median()
print(male_3_median)
#6.여성,3등석
female_3_median=df[(df['pclass']==3) & (df['sex']=='female')]['age'].median()
print(female_3_median)
```
결측치 업데이트할 때 밑에와 같은 방식으로 하면 결측치가 업데이트 되지않음
```py
#isnull 조건을 괄호 안으로 넣어야함,loc사용해야함
df[(df['pclass']==1) & (df['sex']=='male')]['age'].isnull().sum()
df[(df['pclass']==1) & (df['sex']=='male')& df['age'].isnull()]['age'].fillna(male_1_median,inplace=True)
``

```py
#isnull 조건을 괄호 안으로 넣어야함.loc (행,열)
df.loc[(df['pclass']==1) & (df['sex']=='male')& df['age'].isnull(),'age']=male_1_median
df.loc[(df['pclass']==2) & (df['sex']=='male')& df['age'].isnull(),'age']=male_2_median
df.loc[(df['pclass']==3) & (df['sex']=='male')& df['age'].isnull(),'age']=male_3_median

df.loc[(df['pclass']==1) & (df['sex']=='female')& df['age'].isnull(),'age']=female_1_median
df.loc[(df['pclass']==2) & (df['sex']=='female')& df['age'].isnull(),'age']=female_2_median
df.loc[(df['pclass']==3) & (df['sex']=='female')& df['age'].isnull(),'age']=female_3_median
```

# 전처리
### 1.결측치제거
### 2.인코딩(숫자로 바꾸기) =>라벨의 사이즈가 의미값으로 들어갈까봐 원핫인코딩함.
    ### 2-1 -> 정수인코딩(label encoding(기본은 오름차순 정렬)),
    ### 2-2 -> 원핫인코딩 (classification(분류) 원핫인코딩 x, 회귀분석에서는 원핫인코딩 o)

### 3.feature scaling(classification 에선 불필요)
EX) LabelEncoder 예시
```py
#전처리하는 거
from sklearn.preprocessing import LabelEncoder
items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']
encoder=LabelEncoder()
encoder.fit(items)#오름차순으로 정렬함.
labels=encoder.transform(items)#첫번째 데이터를 0번부터 할당함
print('인코딩 변환값',labels)
print('인코딩클래스 fit한 상태 알려줌',encoder.classes_)
print('디코딩 원본 값:', encoder.inverse_transform([4,5,2,0,1,1,3,3]))
```

titanic데이터 LabelEncoder
```py
for i in df1_x:
    encoder=LabelEncoder()
    encoder.fit(df[i])
    df[i]=encoder.transform(df[i])
    print('인코딩 변환값',labels)
    print('인코딩클래스 fit한 상태 알려줌',encoder.classes_)
    #print('디코딩 원본 값:', encoder.inverse_transform([4,5,2,0,1,1,3,3]))
```

Feature scalining:표준화,정규화<br>
이유 : 회귀분석을 할때는 숫자의 크기의 영향을 많이받음<br>
a 피쳐(만단위), b 피쳐(0.1 -0.3 ) 이럴때 회귀 분석에서는 b 피쳐에 가중치가 들어감 a피쳐는 의미를 작게만들어버리고<br>
둘다 백단위로 바꾸어 버리는것임 , 편향되게 하지않게하기위해 <br>
```py
#모듈 :from sklearn.preprocessing import StanardScaler
# scaler = StandardScaler()
def standard_deviation(x):#표준화
    return (x-x.mean())/x.std()
```
```py
#정규화 (백분률로 만듬),최대값 ->1 최소값 ->0(상대평가)
def normalization(x):
    return (x-x.min())/(x.max()-x.min())
```
학습 돌리기위해서 x,y로 데이터 분리
```py
#train test split
X=df.drop('survived',axis=1)
y=df.survived
```
train,test 데이터로 분리
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
```
모델 
```
#머신러닝(random forest ,logisticregression, decision tree classifier)
#decisiontree 성능 향상한게 randomforest, randomforset의 성능이 웬만하면 좋음
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

dt_clf=DecisionTreeClassifier()
#n_job 하드웨어를 건들임, cpu 코어 개수,전체를 다돌리겠다 -1
rf_clf=RandomForestClassifier(n_jobs=-1)
lr_clf=LogisticRegression(n_jobs=-1)


dt_clf.fit(x_train,y_train)
dt_predict=dt_clf.predict(x_test)
rf_clf.fit(x_train,y_train)
rf_predit=rf_clf.predict(x_test)
lr_clf.fit(x_train,y_train)
lr_predict=lr_clf.predict(x_test)


from sklearn.metrics import accuracy_score
print('DecisionTree accuracy score:%.2f:' %accuracy_score(y_test,dt_predict))
print('random forest score :%.2f' % accuracy_score(y_test,rf_predit))
print('logistic score:%.2f' % accuracy_score(y_test,lr_predict))
```

교차검증<br>
### 교차검증(똑같은 데이터를 5번 보는 느낌.)(종류 : kfold,STRATIFIED kfold)

```py
from sklearn.model_selection import KFold
for machine in [dt_clf, rf_clf, lr_clf]:
  scores = []
  kfold = KFold(n_splits=5)
  for i, (train, test) in enumerate(kfold.split(X)):
    X_train = X.values[train]
    X_test = X.values[test]
    y_train = y.values[train]
    y_test = y.values[test]
    machine.fit(X_train, y_train)
    pred = machine.predict(X_test)
    print('RandomForest accuracy score: %.2f' % accuracy_score(y_test, pred))
    scores.append(accuracy_score(y_test, pred))
  print('kfold random forest 평균정확도 %.2f' % np.mean(scores))
  print('-' * 50, '\n')
```

```py
#STRATIFIED kfold
def exec_skfold(machine, X, y, n=5):
  scores = []
  skfold = StratifiedKFold(n_splits=n)
  for i, (train, test) in enumerate(skfold.split(X, y)):
    X_train = X.values[train]
    X_test = X.values[test]
    y_train = y.values[train]
    y_test = y.values[test]
    machine.fit(X_train, y_train)
    pred = machine.predict(X_test)
    print('%s번째 정확도: %.2f' % (i+1, accuracy_score(y_test, pred)))
    scores.append(accuracy_score(y_test, pred))
  print('kfold 평균정확도 %.2f' % np.mean(scores))
```

하이퍼 파라미터 최적화
```py
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth':[1,2,3,4,5,6,7,8,9,10],
              'min_samples_split':[2,3,4,5,6,7,8,9,10],
              'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}
              
clf = DecisionTreeClassifier()
grid_clf = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=5)

import time
start_time = time.time()
grid_clf.fit(X_train, y_train)
print('걸린 시간: ', time.time() - start_time)


```

