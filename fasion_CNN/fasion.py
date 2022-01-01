import numpy as np
import matplotlib.pyplot as plt
from time import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM,Dropout,InputLayer
from tensorflow.keras.layers import Conv2D, MaxPool2D

#하이퍼 파라미터
M_EPOCH = 3
M_BATCH = 300

### 데이터 준비 ###
# 데이터 읽어들이기
# 결과 타입은 numpy의 n-차원 행렬임
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# 4분할된 데이터 형태 출력 확인
print('학습용 입력 데이터 모양 :', X_train.shape)
print('학습용 출력 데이터 모양 :', Y_train.shape)
print('평가용 입력 데이터 모양 :', X_test.shape)
print('평가용 입력 데이터 모양 :', Y_test.shape)

print(X_train[0])
plt.imshow(X_train[0], cmap = 'gray')
plt.show()
print('샘플 데이터 라벨 : ', Y_train[0])

# 샘플 데이터 스케일링 (데이터 전처리) : [0, 1]
X_train = X_train / 255.0
X_test = X_test/ 255.0

#스케일링 후 확인
print(X_train[0])
plt.imshow(X_train[0], cmap = 'gray')
plt.show()

# 채널 정보 추가
# 케라스 CNN 에서 4차원 정보가 필요함 : Tensor 타입임
train = X_train.shape[0] # 60000 면 추출(1차원 배열 : 벡터)
X_train = X_train.reshape(train, 28, 28, 1) # 차원 늘림
test = X_test.shape[0] # 10000
X_test = X_test.reshape(test, 28, 28, 1)

#Tensor로 처리 후 확인
print(X_train[0])
plt.imshow(X_train[0], cmap = 'gray')
plt.show()

#출력 데이터(라벨 정보): One-hot encoding
print('One-hot encoding 전 : ', Y_train[0])

Y_train = to_categorical(Y_train, 10)
print('One-hot encoding 후 : ', Y_train[0])

Y_test = to_categorical(Y_test, 10)

print('학습용 출력 데이터 모양 :', Y_train.shape)
print('평가용 입력 데이터 모양 :', Y_test.shape)

