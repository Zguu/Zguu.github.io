---
title: " [딥러닝] Keras를 활용한 간단한 NN implementation"
tags: DeepLearning NN keras
---
# Keras로 간단한 NN 모델 구축하기

## IMDB movie set Classification
### Hello Data
Keras의 데이터셋 모듈에서 간단하게 데이터를 불러오자. 트레이닝 데이터셋(X), 트레이닝 라벨값(y), 테스트 데이터셋(X), 테스트 라벨값(y)들을 불러온다. 가장 빈번하게 출현하는 단어들 10000개만 데이터에 포함하도록 제한을 걸자. imdb 데이터 객체는 단어와 인덱스를 딕셔너리로 연결한 값들을 제공하며, 이 값들을 이용해 데이터 안에 정수로 표현된 인덱스들을 영화 이름으로 바꿔줄 수 있다. 기본적으로 제공되는 이 딕셔너리는 key가 영화 이름이고, value가 정수 인덱스이다. 우리가 원하는 것은 정수 인덱스에서 영화 이름으로의 변환이므로, 해당 딕셔너리의 key와 value 순서를 바꾸어 저장한다 (reverse_word_index)

```python
from keras.datasets import imdb
import warnings

warnings.filterwarnings('ignore')

(train_data, train_labels), (test_data, test_labels) \
= imdb.load_data(num_words = 10000)

word_index = imdb.get_word_index()
reverse_word_index = dict(
                          [(value, key) for (key, value) in word_index.items()]
                          )
```

위에서 저장한 reverse_word_index 딕셔너리를 활용해서 우리의 데이터가 실제로는 어떤 리뷰값이었는지 확인할 수 있다. 아래의 decoded 결과를 확인하자.
```python
decoded = ' '.join([reverse_word_index.get(i-3,'') \
                    for i in train_data[0]])
decoded
```
decode 된 결과는 다음과 같다.
<center><img src="https://imgur.com/6Njuef0.png" width="80%" height="80%"></center>

### Data Preparation
- 길이가 다른 각 array들을 padding을 활용해 같은 길이의 array로 만들 수 있다. 가장 길이가 긴 array는 길이가 2494개 이므로, 전체 array shape는 (25000, 2494)가 된다. (정수 텐서로 변환된다.) 이후에 이 정수 텐서를 다룰 수 있는 층을 신경망의 첫 번째 층으로 사용한다. (Embedding 층이며, 이에 대한 이야기는 추후에 다룬다.)
- One-hot encoding을 사용한다.
<br>
간단한 One-hot encoder 함수 코드는 다음과 같다.<br>
```python
'''
one-hot encoding
'''

import numpy as np

def vectorize_squences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

X_train = vectorize_squences(train_data)
X_test = vectorize_squences(test_data)
```
y 값들은 다음과 같이 변환해준다.
```python
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
```
## Simplest NN model implementation
세상 간단한 NN 모델 구축의 형태
- Two Hidden layers
- Output layers
- Optimization function은 여러개가 존재하지만, ```rmsprop``` 최적화 함수를 대부분의 상황에서 사용해도 된다고 한다. (굳!)

```python
'''
implementing model.......
'''
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu',
                       input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

X_val = X_train[:10000]
partial_X_train = X_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_X_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (X_val, y_val))
```

위에서 간단한 형태의 NN 구조에 데이터를 입력해 훈련을 진행했고, 해당 훈련에 의한 training loss와 validation loss는 아래 figure에서 확인할 수 있다.
```python
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure(figsize = (10,8))
plt.plot(epochs, loss, 'bo', label = 'Training loss',
         color = 'r')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss',
         color = 'c')
plt.title('Training and validation loss', fontsize = 14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```
<center><img src="https://imgur.com/82fhBzz.png" width="80%" height="80%"></center>
Training Set에서 loss는 Epoch이 증가하면 증가할수록 계속 낮아지는 모습을 보인다. 반면에, Epoch이 약 3이 되는 시점에서 Validation loss는 더이상 감소하지 않고 오히려 미세하게 증가하는 모습을 보이며, 이는 Over-fitting 이라고 볼 수 있다. 이와 같은 추세는 아래의 accuracy 그래프에서도 나타난다.
```python
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.figure(figsize = (10,8))
plt.plot(epochs, acc, 'bo', label = 'Training accuracy',
         color = 'r')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy',
         color = 'c')
plt.title('Training and validation accuracy', fontsize = 14)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()
```
<center><img src="https://imgur.com/0ItLvUY.png" width="80%" height="80%"></center>
Loss 그래프에서도 확인 할 수 있었지만, accuracy 또한 3,4회 이후 epoch에서 감소하는 추세이며, 최종적인 validation set에서의 정확도는 0.8보다 약간 큰 결과이다.
## Reuter News Multiclass Classification
- 위의 예시에는 0 아니면 1 값만 갖는 분류 문제였지만, 이번에는 총 46개의 클래스로 분류를 진행하자.
- 여러개 범주로 분류될 수는 있지만, 결국 하나의 값만 갖는 결과가 나오기 때문에 정확히 말하면 single-label multiclass classification에 해당한다.
- 만약 각각의 데이터 포인트가 여러 개의 범주에 속할 수 있다면, multi-label, multiclass classification에 해당한다.

### Hello Data
1986년 로이터가 공개한 **로이터 데이터셋** 을 사용하자. 총 46개의 토픽이 있으며 짧은 뉴스 기사에 대한 데이터들을 포함한다. 로이터 뉴스 데이터는 위의 IMDB 데이터와 거의 완벽하게 같다. 입력 데이터 X는 단어들로 이뤄져있다. 다만, 딱 한가지 다른 점은 출력 값 (뉴스 카테고리에 해당하는 값)은 0 또는 1의 형태를 취하는 것이 아니라, 어떤 카테고리의 뉴스에 속하는지를 알려주는 라벨이며 이 라벨이 총 46개이기 때문에, 총 46가지의 출력 종류가 존재한다. 따라서 트레이닝 라벨과 테스트 라벨들의 값을 카테고리컬한 값으로 코딩해줘야한다. 이 부분만 앞의 IMDB의 binary Classification과 다를 뿐이다. 아래의 데이터 로딩 및 디코드 예시는 IMDB 예시와 완벽히 같다.
```python
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = \
reuters.load_data(num_words = 10000)

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, keys) for (keys, value) in
                          word_index.items()])

decoded_ex = ' '.join([reverse_word_index.get(i-3, '') \
                       for i in train_data[0]])
decoded_ex
```
<center><img src="https://imgur.com/cogPsWg.png" width="80%" height="80%"></center>

마찬가지로 트레이닝 데이터와 테스트 데이터의 단어 셋들을 모두 one-hot encoding을 진행해준다.
```python
import numpy as np

def vectorize_sequence(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

X_train = vectorize_sequence(train_data)
X_test = vectorize_sequence(test_data)

'''
vectorize label
'''
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
```

### Model Implementation
Binary Classification이 아니고 Multi Class Classification 에 해당하므로, 컴파일에서 손실함수는 categorical_crossentropy 로 지정한다. 또한, 최종 아웃풋 레이어에서 acitvation 함수도 softmax 함수로 지정해준다. 이 두가지 변경점을 제외하면 역시 이전 예시에서 레이어 구성과 동일하다.

```python
'''
softmax for multiclass probabilistic prediction
'''

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu',
                       input_shape = (10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

X_val = X_train[:1000]
partial_X_train = X_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
```

컴파일 이후 fit 과정에서, Validation 데이터 셋들을 지정해주고, 학습 과정에서의 accuracy, loss들을 history 변수에 저장해주자. 이후에 loss, accuracy 추이 그래프를 그리는 데에 유용하게 사용한다.
```python
history = model.fit(partial_X_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (X_val, y_val))
```

loss 추이와 accuracy 추이는 아래의 두 그래프에서 확인할 수 있다.

```python
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure(figsize = (10, 8))
plt.plot(epochs, loss, 'bo', label = 'Training loss',
         color = 'r')
plt.plot(epochs, val_loss, 'b', label = 'validation loss',
         color = 'c')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```
<center><img src="https://imgur.com/7HYNP5r.png" width="80%" height="80%"></center>

```python
loss = history.history['accuracy']
val_loss = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.figure(figsize = (10, 8))
plt.plot(epochs, loss, 'bo', label = 'Training accuracy',
         color = 'r')
plt.plot(epochs, val_loss, 'b', label = 'validation accuracy',
         color ='c')
plt.title('training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```
<center><img src="https://imgur.com/KZdMBBR.png" width="80%" height="80%"></center>

역시나, 각 훈련시에 over-fitting이 일어나는 시점을 확인할 수 있으며, 이에 대한 설명은 위의 예시와 완벽히 같으므로 생략한다.

## House pricing prediction
마지막으로, 앞의 두 예시에서 확인한 분류 문제가 아니라 회귀 문제에 NN 모델을 적용해본다. 대부분 모델 구성은 같으므로 중복 부분은 설명하지 않는다.
### Hello Data
Data 전처리에서 단 한가지 유념해야 할 것은, 단위에 민감한 회귀 문제이므로, 각 변수들을 정규화해준다. 정규화를 하지않아도 뉴럴넷이 자동으로 학습하는 경우도 많이 있지만,단위가 큰 변수에 영향을 크게 받아서 해당 변수에만 의존해 경사 하강을 진행할 가능성이 높아지므로 변수 정규화를 진행한다. 회귀 문제에 해당하므로, ```softmax```나 ```sigmoi```와 같은 출력층 함수는 사용할 필요가 없다. loss 함수는 회귀 문제이므로 ```mse``` 를 사용하자.
```python
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = \
boston_housing.load_data()

'''
test data를 정규화할 때도
train data에서 얻은 mean, std 사용한다.
test data에서 얻은 어떠한 정보, 힌트도 사용하지 않는다.
'''

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean)/std
test_data = (test_data - mean)/std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu',
                           input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
```

### K-Fold cross validation
일반적인 머신러닝의 validation 과정과 마찬가지로, K-fold validation을 진행한다. 각 Fold에서의 퍼포먼스를 종합하여 평균적으로 모델의 퍼포먼스를 구한다.
K-fold CV 과정에 대한 알고리즘은 아래와 같다. (굳이 직접 알고리즘 작성해보기)
```python
'''
K-fold-cv 함수 직접 짜볼라했는데 오류난다 하
다음에 다시 ㄱㄱ
'''

def k_fold_cv(train_data, train_targets, k = 5,
              num_epochs = 500, batch_size = 1):

    fold_size = len(train_data) // k
    mse_arr = []
    #mae_arr = []

    all_mae_history = []
    for i in range(0,k):
        print('처리중인 폴드......{}'.format(i))

        val_data = train_data[i*fold_size : (i+1)*fold_size]
        val_targets = train_targets[i*fold_size : (i+1)*fold_size]

        partial_train_data = np.concatenate([train_data[:i*fold_size],
                                     train_data[(i+1)*fold_size:]], axis=0)

        partial_train_targets = np.concatenate([train_targets[:i*fold_size],
                                        train_targets[(i+1)*fold_size:]],
                                        axis=0)        

        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data = (val_data, val_targets),
                            epochs = num_epochs, batch_size = 1,
                            verbose = 0)

        mae_history = history.history['val_mae']

        #mae_arr.append(val_mae)
        all_mae_history.append(mae_history)

    return(all_mae_history)     

all_mae_history = k_fold_cv(train_data, train_targets)
```
각 K번째 fold에서 학습을 진행하고 해당 과정들에서 결과들을 모두 history로 저장했다. 이를 그래프로 그려 확인하면 다음과 같다.
```python
plt.figure(figsize = (12,8))
plt.title('MAE vs epochs', fontsize = 15)
plt.xlabel('epochs')
plt.ylabel('MAE')
plt.plot(range(0,500), np.array(all_mae_history).mean(axis=0),
         color = 'c')
plt.show()
```
<center><img src="https://imgur.com/hYYanU7.png" width="80%" height="80%"></center>
위의 MAE 추세 그래프에서 확인할 수 있듯이, epoch이 지남에 따라 초반에 급격하게 감소했던 MAE가 점차 미세량 증가해 일정 수준에서 진동하고 있음을 알 수 있다. 하지만 어느 epoch 수준이 적절한지 그래프에서 확인할 수 없으므로, 해당 그래프를 조금 더 확대해보거나 다르게 관찰해볼 필요가 있다.<br>
$\ $EMA 방법론을 활용해서, 해당 그래프의 움직임을 다시 관찰한다. EMA는 과거 데이터 값에 weight를 누적시키며 바라보므로, 그래프가 조금더 smooth 해진다. 최초 10 epoch 정도는 명백히 MAE가 줄어드는 것 같으므로, EMA 과정에서 제외한다. EMA 함수는 아래와 같이 정의한다.
```python
def Exponential_Moving_Average(arrs, alpha = 0.1):
    '''
    arrs = input array
    alpha = past data accumulating coefficient
    '''    
    EMA_arr = []

    for i, val in enumerate(arrs):
        if EMA_arr:
            EMA = arrs[i-1]*alpha + arrs[i]*(1-alpha)
            EMA_arr.append(EMA)
        else:
            EMA_arr.append(val)

    return(EMA_arr)  
```

이를 이용해 Smoothed MAE 추세를 살펴보면 아래와 같다.
```python
avg = np.array(all_mae_history).mean(axis=0)
smoothed = Exponential_Moving_Average(avg[10:])

plt.figure(figsize = (12,8))
plt.title('Smoothed MAE', fontsize = 15)
plt.xlabel('epochs')
plt.ylabel('Smooted MAE')
plt.plot(range(10,500), smoothed, color = 'c')
plt.show()
```
<center><img src="https://imgur.com/NfyaRxD.png" width="80%" height="80%"></center>
epoch이 약 60을 초과하면서부터 MAE는 명백히 over fitting을 가리키고 있다. 해당 추세는 smoothed 되기 전 원래 그래프에서는 확인하기가 힘들었던 것이다. EMA를 활용해 시각적으로 적절 epoch을 찾을 수 있게 됐다. 물론, 증가 감소 추세의 극점을 찾는 함수로 발견할 수도 있다. <br>
이상으로 간단한 NN 모델을 활용해 분류와 회귀 문제에 적용해보았다. 끗!
