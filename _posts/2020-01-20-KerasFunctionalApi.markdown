---
title: " [딥러닝] Keras 함수형 API"
tags: DeepLearning
---

이 포스팅은 ```케라스 창시자에게 배우는 딥러닝 Chapter 7``` 의 내용 중 일부를 발췌하여 정리한 것입니다.
# 함수형 API?
$\ $케라스를 사용할 때, ```Sequential()``` 모델을 이용해서 빠르고 효율적으로 신경망을 구축할 수 있다. 하지만, ```Sequential()``` 모델만을 사용해서는 조금 더 복잡한 그래프 형태를 취하는 신경망을 모델링할 수 없다. ```Sequential()``` 만을 이용하면 단일 데이터 입력 소스로, 단일 출력 소스를 내보낼 수 밖에 없다. 아래 그림처럼 ```Sequential API```는 순서대로 층을 쌓아 만들기 때문에, 다양한 입력 소스에서의 데이터를 다양한 출력으로 내보내기 힘들다. 반면에, 함수형 API는 이를 가능하게 한다. 아래 그림에서 확인할 수 있는 것 처럼, 출력 상황에서 branch가 가능하며, 거꾸로 입력을 다양한 소스에서 받아올 수 있다.
<center><img src="https://imgur.com/WdG4yD2.png" width="80%" height="80%"></center>

## 함수형 API 코드 예시
$\ $함수형 API는 입출력 텐서를 정의하고, 우리가 정의한 층을 사용해 입출력이 드나들게 된다. 간단한 코드 구현 예시는 아래와 같다.
```python
from keras import Input, layers

input_tensor = Input(shape = (32,))
dense = layers.Dense(32, activation = 'relu')
output_tensor = dense(input_tensor)
```
우리에게 친숙한 ```Sequntial()``` 모델과 함수형 API를 비교해보자.

```python
from keras.models import Sequential, Model
from keras import layers
from keras import Input

## Sequential() API
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation = 'relu', input_shape = (64,)))
seq_model.add(layers.Dense(32, activation = 'relu'))
seq_model.add(layers.Dense(10, activation = 'softmax'))

## Funtional API
input_tensor = Input(shape = (64,))
x = layers.Dense(32, activation = 'relu')(input_tensor)
x = layers.Dense(32, activation = 'relu')(x)
output_tensor = layers.Dense(10, activation = 'softmax')(x)

model = Model(input_tensor, output_tensor)

model.summary()
```
함수형 API는 입출력 텐서만을 사용해서 모델 객체를 간단히 만들 수 있다. 그 과정에서 필요한 모든 층들은 케라스가 알아서 생성하게 되며 이를 연결해서 전체적인 신경망을 완성한다. 위에서 생성한 함수형 API의 summary는 아래와 같다.
<center><img src="https://imgur.com/lBW3WJI.png" width="80%" height="80%"></center>
