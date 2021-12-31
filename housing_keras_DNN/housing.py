import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#하이퍼 파라미터
MY_EPOCH=500
MY_BATCH=64
heading = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']
raw = pd.read_csv('housing.csv')

print(raw.head(10))
print(raw.describe())

#Z-점수 정규화
#결과는 numpy의 n-차원 행렬 형식
scaler = StandardScaler()
Z_data = scaler.fit_transform(raw) # 컬럼 라벨 없는 데이터로만 구성된 행렬

# matrix -> DataFrame
Z_data = pd.DataFrame(Z_data, columns = heading)
# Z_data = raw
#정규화된 데이터 출력 확인
print('정규화된 데이터 샘플 10개:\n', Z_data.head(10))
print('정규화 된 테이터 통계:\n', Z_data.describe())

#데이터를입력과출력으로분리
print('\n 분리전 데이터 모양:', Z_data.shape)
X_data = Z_data.drop('MEDV', axis = 1)
Y_data = Z_data['MEDV']

#데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.3)

print('학습용 입력 데이터\n', X_train.shape)
print('학습용 출력 데이터\n',Y_train.shape)
print('평가용 입력 데이터\n',X_test.shape)
print('평가용 출력 데이터\n',Y_test.shape)

# Z_data 시각화
# sns.set(font_scale=1)
# sns.boxplot(data = Z_data, palette='dark')
# plt.show()

### 인공 신경망 구현 ###

# 케라스 DNN구현
model = Sequential()
input = X_train.shape[1] # 입력층 뉴런 12개 지정

model.add(Dense(200, input_dim= input, activation='relu'))
model.add(Dense(1000, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(500, activation = 'relu'))
model.add(Dense(1))

print('\nDNN 요약')
model.summary()

### 인공 신경망 학습 ###

#최적화 함수 (가중치 보정)와 손실 함수 (오차율) 지정
model.compile(optimizer='sgd', loss = 'mse')

print('\nDNN 딥러닝 학습 시작')
begin = time()

model.fit(X_train, Y_train, epochs= MY_EPOCH, batch_size=MY_BATCH, verbose = 2)

end = time()
print('총 딥러닝 학습 시간 : {:.1f}초'.format(end-begin))

### 인공 신경망 모델 평가 및 활용 ###

# 신경망 모델 평가 및 손실값 계산
loss = model.evaluate(X_test, Y_test, verbose = 0)
print('\nDNN 평균 제곱 오차(MSE):{:.2f}'.format(loss))

#신경망 활용 및 산포도 출력
pred = model.predict(X_test)
sns.regplot(x=Y_test, y = pred) #회귀선 그래프

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

#주택가격(MEDV) 비교
print(pd.DataFrame(pred).head(10))