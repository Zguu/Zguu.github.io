---
title: " [머신러닝] 추천 시스템에 SVD를 사용하는 이유"
tags: MachineLearning Recommender System SVD LinearAlgebra
---

# SVD를 다시 떠올려보자
$\ $SVD를 이용하면, 우리에게 주어진 어떤 행렬 $A$를 총 세개의 행렬로 분해할 수 있다. <br>
<center>$$A = U\sum V$$ </center>
이렇게 분해된 총 세개의 행렬을 다시 곱하면 당연히도 원래 행렬 $$A$$를 얻게된다. SVD 행렬 분해는 일반적으로 non-sparse 행렬에서는 매우 잘 작동하지만, real data 들이 공통적으로 보여주는 spare 형태 행렬에서는 잘 작동하지 않는다
## Latent Factor??
$\ $ Latent factor는 실제로 우리에게 주어진 데이터에서는 가시적으로 보이지 않는 값들이다. 예를 들어보자. 한 유저가 인터스텔라, 인셉션에는 각각 평점 9점을 줬고, 타이타닉이랑 로미오와 줄리엣에는 2점을 줬다고 가정해보자. 우리가 알고있는 이 정보 뿐이지만, 우리는 이 유저가 SF영화는 좋아하고 로맨스 영화는 별로 안좋아하는 것 같다는 짐작을 해볼 수 있다 (물론 데이터가 좀 적긴하지만..). 이러한 짐작이 맞을 지 틀릴 지는 모르지만, 분명 영화 평점을 주는 데에 있어서 영향을 줄 수 있는 변수들이다. 하지만 SF영화 또는 로맨스 영화 라는 데이터가 처음부터 우리에게 주어졌는가? 라고 하면 그렇지 않았다. 즉, 겉으로 드러나지 않는 (추측해볼 수 있는 기저에 깔린 변수들) 변수들을 Latent factor로 볼 수 있다.

## funk SVD?
$\ $Simon Funk 에 의해 개발된 funkSVD는, sparse 행렬에서 우리가 알고있는 값들만 이용해서 행렬 분해를 진행한다. 일반적으로 SGD 최적화 방법론을 사용하며, 이 행렬 분해를 통해 Latent Factor를 계산해낸다. 아래 에시 데이터들을 보면서 차근차근 funkSVD의 계산 아이디어를 확인해보자.

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
<center>New value = $$ 0.8 + 0.1\cdot 2(9 + 0.07)\cdot (-0.2) = 0.44$$ </center>
이와 같은 방법으로 모든 값들을 업데이트 해주면, $$0.44, 1.38, 0.053$$의 값을 얻는다. 이를 이용해 새로운 $U$를 얻는다.
<center><img src="https://imgur.com/ublFO8X.png" width="80%" height="80%"></center>
새롭게 얻은 $U$를 활용해 $V$ 또한 업데이트를 진행해주면 아래와 같다.
<center><img src="https://imgur.com/sd9lwpH.png" width="80%" height="80%"></center>
이에 대한 값들을 원래 $$U, V^T$$ 행렬들에 입력하여 마무리한다.
<center><img src="https://imgur.com/hIkRnKu.png" width="80%" height="80%"></center>
<center><img src="https://imgur.com/7cN7e9M.png" width="80%" height="80%"></center>





> reference
  https://medium.com/datadriveninvestor/how-funk-singular-value-decomposition-algorithm-work-in-recommendation-engines-36f2fbf62cac
