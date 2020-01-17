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
간단한 One-hot encoder 함수 코드는 다음과 같다.
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
- Optimization function은 여러개가 존재하지만, ```rmsprop```을 대부분의 상황에서 사용해도 된다고 한다. (굳!)

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
