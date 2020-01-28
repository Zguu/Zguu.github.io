---
title: " [머신러닝] 추천 시스템과 SVD"
tags: MachineLearning Recommender System SVD LinearAlgebra
---

# SVD를 다시 떠올려보자
$\ $SVD를 이용하면, 우리에게 주어진 어떤 행렬 $A$를 총 세개의 행렬로 분해할 수 있다. <br>
<center>$$A = U\sum V^T$$ </center>
이렇게 분해된 총 세개의 행렬을 다시 곱하면 당연히도 원래 행렬 $$A$$를 얻게된다. SVD 행렬 분해는 일반적인 non-sparse 행렬에서뿐만 아니라 real data 들이 공통적으로 보여주는 spare 형태 행렬에서도 잘 작동한다. 하지만, 영화 평점을 예측하는 것과 같은 explicit dataset이 아닌, implicit dataset 에서의 추천에서는 잘 작동하지 않을 수 있다.
> 잘 작동한다? 라는 말은, 분해된 세개의 행렬을 다시 곱해서 원래 행렬 $A$를 만들었을 때, 원래 행렬과 값 차이가 거의 나지 않는다는 말이다. 하지만 원래 행렬 $A$가 sparse하면 할 수록, 분해 이후 얻게 된 행렬들의 곱으로 다시 $A$를 만들어도 값이 일치하지 않을 수 있다.

## Latent Factor??
$\ $ Latent factor는 실제로 우리에게 주어진 데이터에서는 가시적으로 보이지 않는 값들이다. 예를 들어보자. 한 유저가 인터스텔라, 인셉션에는 각각 평점 9점을 줬고, 타이타닉이랑 로미오와 줄리엣에는 2점을 줬다고 가정해보자. 우리가 알고있는 이 정보 뿐이지만, 우리는 이 유저가 SF영화는 좋아하고 로맨스 영화는 별로 안좋아하는 것 같다는 짐작을 해볼 수 있다 (물론 데이터가 좀 적긴하지만..). 이러한 짐작이 맞을 지 틀릴 지는 모르지만, 영화의 카테고리나 장르는 분명 영화 평점을 주는 데에 있어서 영향을 줄 수 있는 것들이다. 하지만 SF영화 또는 로맨스 영화 라는 데이터가 처음부터 우리에게 주어졌는가? 라고 하면 그렇지 않았다. 즉, 겉으로 드러나지 않는 (추측해볼 수 있는 기저에 깔린) 변수들을 Latent factor로 볼 수 있다.<br>
$\ $일반적으로, Matrix Factorization techniques 들을 사용할 때에, Latent Factor의 갯수는 너무 크지도 적지도 않게 잡는 것이 일반적이다. 너무 큰 값으로 Latent factor 수를 잡으면 당연하게도 over-fitting이 일어나고, 너무 적은 값의 Latent factor 수는 너무 큰 error를 발생시킬 수 있다. 너무 크지도 않고 적지도 않다는 '적당한' 이라는 말은 매우 추상적이고 쓸모없는 말처럼 들리는데 명확한 기준이 없으니 ovefitting 이 일어나지 않는 선에서 해석 가능한 Latent factor 수를 잡는게 맞지 않나 싶다. 이에 대한 구체적인 이야기는 다른 포스팅에서 다뤄보기로 하자.
> 아마 Latent factor로 가질 수 있는 가장 큰 값은 그 행렬의  Column Rank 이지 않으려나? 물론 그렇게까지 Latent factor를 키울 일은 많이 없을 것이다. 이건 잘 모르겠네.

## funk SVD?
$\ $Simon Funk 에 의해 개발된 funkSVD는, sparse 행렬에서 우리가 알고있는 값들만 이용해서 행렬 분해를 진행한다. 일반적으로 SGD 최적화 방법론을 사용하며, 이 행렬 분해를 통해 Latent Factor를 계산해낸다. 아래 에시 데이터들을 보면서 차근차근 funkSVD의 계산 아이디어를 확인해보자. (사실 계산 아이디어 라기보다는... 아무런 값이나 우선 넣고 해당 값들을 점점 업데이트 해나가는 과정에 불과하다.)

첫번째로, 우리에게 주어진 User-Item 행렬은 다음과 같다. 역시나 sparse matrix 형태를 보여준다.

<center><img src="https://imgur.com/VpXVoZV.png" width="80%" height="80%"></center>
우리는 이 행렬을 $$U$$, $$V^T$$ 행렬들(Latent Factors)로 분해할 것이다. 처음에 우리는 이 행렬들을 채울 값을 알지 못하기 때문에, Latent factor의 갯수 정도만 사전에 설정해주고, 행렬 안의 값들은 random value들로 채워준다. 아래는 User에 대한 Latent factor $$U$$이며, random 값으로 채워져있다.
<center><img src="https://imgur.com/BfWyb37.png" width="80%" height="80%"></center>
아래는 Item에 대한 Latent factor $$V^T$$이며, 역시 처음이라 random 값으로 채워져있다. 현재 User와 Item에 대한 Latent factor 갯수는 3개로 잡혀있다.
<center><img src="https://imgur.com/tqkb7SM.png" width="80%" height="80%"></center>
$\ $맨 위의 User-Item 행렬에서 1번 유저가 3번 품목에 대해 매긴 점수는 9점임을 확인할 수 있다. 이 9점은 **실제 값** 에 해당한다. 위에서 두번째 유저의 Latent factor 행렬에서 1번 유저의 Latent factor vector는 $$(0.8, 1.2, -0.2)$$ 임을 확인할 수 있다. 또한, 맨 아래에 있는 Item Latent factor 행렬에서는 3번 품목의 Latent factor vector가 $$(-0.2, 0.1, 0.14)$$ 임을 확인할 수 있다. 이 두 벡터들의 내적 ($$uv^T$$)은 예측된 값으로 볼 수 있다. 계산 결과는 다음과 같다. $$(0.8\cdot -0.2) + (1.2\cdot 0.1) + (-0.2\cdot 0.14) = -0.07 $$. 실제 값은 9점이었는데, 예측한 값은 -0.07이라니.. 택도 없는 결과지만 실망할 필요는 없다. 왜냐면 우리가 당연히 처음에 Latent factor matrix들을 랜덤한 값으로 마구잡이로 채워넣었으니 말도 안되는 예측 값이 나오는 건 당연하다. 우리는 여기서 실망하지 말고, 이 예측값이 실제값에서 벗어난다는 점에 입각하여 Latent factor matrix들의 값들을 수정해나가자. Gradient Descent를 활용해서 error를 최소화해나가는 과정은 아래와 같다. <br>
<center>$$U(i)\  or\  V(i) + \alpha\cdot(actual - predicted) \cdot V(i)\  or\  U(i) $$</center>
$\alpha$는 learning rate에 해당하며, $$U(i), V(i)$$는 이전에 설정한 Latent factor 행렬들의 랜덤한 값들이다. 우리가 앞에서 선택한 상황 (1번 유저의 3번 품목에 대한 rating 및 latent factors)의 데이터들을 정리해서 다시 써보면 아래와 같다.
<center><img src="https://imgur.com/478vKnb.png" width="80%" height="80%"></center>
$\ $첫번째 행 첫번째 열의 데이터 0.8을 0.1의 learning rate($\alpha$)로 업데이트 해보자. <br>
<center>$$New value = 0.8 + 0.1\cdot 2(9 + 0.07)\cdot (-0.2) = 0.44$$ </center>
이와 같은 방법으로 모든 값들을 업데이트 해주면, $$0.44, 1.38, 0.053$$의 값을 얻는다. 이를 이용해 새로운 $U$를 얻는다.
<center><img src="https://imgur.com/ublFO8X.png" width="80%" height="80%"></center>
새롭게 얻은 $U$를 활용해 $V$ 또한 업데이트를 진행해주면 아래와 같다.
<center><img src="https://imgur.com/sd9lwpH.png" width="80%" height="80%"></center>
이에 대한 값들을 원래 $$U, V^T$$ 행렬들에 입력하여 마무리한다.
<center><img src="https://imgur.com/hIkRnKu.png" width="80%" height="80%"></center>
<center><img src="https://imgur.com/7cN7e9M.png" width="80%" height="80%"></center>

## SGD (Stochastic Gradient Descent)
$\ $위의 과정을, SGD의 개념을 이용해 파이썬으로 구현해보자. SGD는 위에서 봤듯이, 각 cell들을 업데이트 하는 과정을 반복하고, 예측 값과 실제 값 사이의 오류 (RSS or something...)들을 줄여나간다. 이에 대한 코드를 numpy를 이용해서 작성해보고 구현해보자.
> 많은 힌트를 surprise 패키지의 class 소스 코드에서 얻었다.

```python
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

n_factors = 10

init_mean = 0
init_std_dev = 0.1

n_user = 30
n_item = 30

## r = actual rating
r = np.random.randint(0,6,(n_user,n_item))

pu = np.random.normal(init_mean, init_std_dev, (n_user, n_factors))
qi = np.random.normal(init_mean, init_std_dev, (n_user, n_factors))
```
잠재 변수의 수는 10개, 현재 유저 수는 30명, 총 아이템은 30개가 있다고 가정하고 진행한다. 1과 5사이의 정수 값을 score로 하는 매트릭스 $R$을 임의로 생성한다. 이후에 해당 매트릭스 $R$을 $$p_u, q_i$$로 분해한다. 우리가 첫 행렬을 30 by 30 으로 설정했고, 잠재 변수의 크기는 10이므로 $$p_u$$ 행렬은 30 by 10, $$q_i$$ 행렬은 10 by 30 행렬로 분해된다.
```python
u = 1
i = 1

r_hat = 0

lr_all = .05

lr_pu = lr_all
lr_qi = lr_all

lamb = 0.01

def learn_cell(x,y, epochs = 1):
    u = x
    i = y

    rss = 0
    for k in range(0,epochs):
        r_hat = 0
        r_hat = np.dot(pu[u], qi[i])

        err = r[i,u] - r_hat

        for f in range(n_factors):
            puf = pu[u, f]
            qif = qi[i, f]

            pu[u, f] += lr_pu * (err * qif - 0.02 * puf)
            qi[i, f] += lr_qi * (err * puf - 0.02 * qif)

    return(pu, qi, r_hat)
```
$$u, i$$는 각각 1로 설정했는데, 이는 $R$ 행렬에서 첫 번째 행, 첫 번째 열에 해당하는 $r$ 값을 에측하도록 먼저 행렬분해를 진행하겠다는 것이다. 이후 학습률 `lr, lambda` 와 같은 변수들을 지정해주고 SGD 함수를 정의한다.
```python
r_predict = np.zeros((n_user, n_item))

rss_arr = []
epochs = 5

for x in range(0,n_item):
    for y in range(0,n_user):
        r_predict[y,x] = round(learn_cell(x,y,epochs)[2],2)
        puf, qif = learn_cell(x,y,epochs)[0:2]
        rss = sum(sum((r_predict - r)**2))

        rss_arr.append(rss)
print(rss)
```
해당 함수가 모든 cell들의 score를 훑고 지나가며, 학습을 진행할 때 ```RSS``` 값을 벡터로 기록한다. 사실, 하나의 셀들에 대해 한번씩 훈련을 하고 전체 행렬의 모든 셀을 한 번 훑은 것을 `epoch = 1` 로 보는게 조금 더 자연스러워 보이지만... 함수를 짜다보니 이렇게 되버렸다. 결과에는 큰 차이가 없다. 정답에 해당하는 $$R$$ 행렬의 10개 행과, 우리가 예측한 $$R_{predict}$$ 행렬의 10개 행렬을 아래와 같이 확인해보자. `jupyter`에서 한 번에 확인할 수 있는 column의 숫자는 아래와 같이 지정할 수 있다.
```python
pd.set_option('display.max_columns', 20)

pd.DataFrame(r).head(10)
pd.DataFrame(r_predict).head(10)
```
<center><img src="https://imgur.com/OqXrWD5.png" width="80%" height="80%"><img src="https://imgur.com/nxIxuHw.png" width="80%" height="80%"></center>
$\ $행렬의 사이즈가 크지 않은 덕분에(?), 몇 번의 반복된 학습 이후로 빠르게 정답을 잘 찾아내고 있다.

각 ```score```들을 학습하는 ```epoch``` 수가 증가할 수록 빠르게 ```RSS```가 줄어듦을 확인할 수 있다. 아래 그래프는 아주 당연하게도 학습되는 cell의 수가 늘어날 수록 `RSS` 값이 선형적으로 감소한다는 것을 보여준다.
<center><img src="https://imgur.com/wN1JMM8.png" width="80%" height="80%"></center>

> reference
  https://medium.com/datadriveninvestor/how-funk-singular-value-decomposition-algorithm-work-in-recommendation-engines-36f2fbf62cac
