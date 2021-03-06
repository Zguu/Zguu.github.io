---
title: " [머신러닝] Gaussian Processes"
tags: MachineLearning
---

# Gaussian Process (GP)
$\ $일반적인 머신러닝 알고리즘들은 트레이닝 데이터 셋이 주어졌을 때, 해당 데이터들에 최고로 잘 맞는 ("best fit") 하나의 함수를 찾는 것을 목표로 한다. 이 함수를 이용해서 미래의 데이터 입력 값이 들어오면 출력 값에 대한 함수 값을 계산할 수 있다. 지금까지의 이러한 방법에서 조금은 접근이 다를 수 있는 ***Bayesian methods*** 에 대해 살펴보자. 전통적인 학습 알고리즘들과는 다르게, 베이지안 알고리즘은 최적의 함수 fitting을 찾는 것으로 볼 수는 없다. 이 방법론은 모델들에 걸쳐 존재하는 사후 분포를 계산한다. 이러한 분포들은 모 측정에 대한 우리의 불확실성을 정량화 하는 유용한 방법을 제시하며, 새로운 데이터셋이 들어올 때 robust한 예측을 가능하도록 한다. <br>
$\ $이러한 베이지안 회귀 알고리즘들 중에, kernel-based fully Bayesian regression algorithm 은 Gaussian process regression으로 불려진다. 전통적인 통계학 분야가 아닌 공학 분야에 있어서 GP의 명확한 장점들은 다음과 같다.

- 예측 값들에 대한 신뢰 구간을 유연하게 표현할 수 있다.
- 트레이닝 데이터들을 interpolate한다.
- 미분가능한 비선형 모델을 생성한다.
- 생성된 회귀 모델을 바탕으로 실험 디자인과 최적화를 진행할 수 있다.

특히, 공학 분야에서 생성되는 데이터들은 우리가 흔히 부르는 **빅데이터** 와는 다르게 전형적인 grid 형태의 데이터 형태를 보여주는데, GP는 이러한 grid 형태 데이터를 사용해서도 좋은 추론 결과를 이끌어낼 수 있다. 아래의 이미지와 문구가 GP 개념에 대한 직관적 이해를 도울 수 있다.

<center><img src="https://imgur.com/ANWhZBL.png" width="80%" height="80%"></center>

---
<center>***Distributions not just over random vectors but in fact distributions over random functions***</center>

## Gaussian Process Motivation
$\ $ 가우시안 프로세스를 간단하게 2차원 공간에서 생각해보자. 우리는 현재 1개의 데이터를 갖고 있다 ($$x_1, y_1$$). 이후에, 다른 $$x_n$$ 에서는 어떤 함숫값 $$y_n$$ 를 갖는지를 알아내고자 한다. GP는 이 함숫값 $$y_n$$들이 갖는 값을 점으로 나타나는 것이 아니라 분포로 나타내고자 하며, 이 분포가 Gaussian 분포를 따르기 때문에, 해당 분포의 평균과 분산을 이용해서 표현한다.
> 예를 들어보자, $x$ 가 [-6, 6] 범위에서 존재하고 있고, 우리는 이 전체 범위에서 모든 $x$ 값은 평균적으로 0일 것이라고 믿고있다. 하지만 우리의 이런 믿음이 틀릴 확률도 있는데, 우리는 모든 $x$에서 평균적으로 표준편차가 1이라고 생각한다. 즉, 모든 $x$들에서 함숫값은 $$-3 (-3\sigma) 에서 +3 ( 3\sigma)$$ 에 존재할 확률이 99%라는 것이다.

여기서 특이한 점은, 우리가 이미 알고있는 $$x_1$$ 에서의 함숫값 $$y_1$$ 이용해서 $$x_n$$ 에서의 $$y_n$$을 표현할 때에, $$x_1$$과 $$x_n$$ 간의 공분산 (covariance)를 이용해서 표현하며, 이 공분산은 kernel 이라고도 부른다. 이에 대한 직관적인 이해가 힘들 수도 있는데, $$x_1$$과 $$x_n$$ 간에 공분산이 존재한다는 것은, 우리가 관측한 $$x_1$$ 값 주변에 있는 $$x_n$$ 값일수록, 그에 대한 $$y_n$$ 추정이 더욱 정확할 것이라는 의미를 함축해야 한다. 따라서, 우리는 공분산관계를 사용하는 것이다. 또한 가까운 거리에 있는 $$x$$ 값들일수록 공분산 값이 높아야함므로, 일반적으로 distance measure 기반의 커널들을 공분산 커널로 사용한다.
> 위의 상황에서 연장하여, 우리에게 추가적인 데이터가 주어진 상황을 가정해보자. 우리에게 $$x = 1$$ 에서 $$y = 1$$ 이라는 추가적인 데이터 셋이 주어졌다. 즉, 우리는 지금까지 모든 $$x$$에서 어느정도의 에러를 허용하는 선에서 평균적으로 함숫값은 전부 다 0 일거야 라고 '찍고' 있었는데, 갑자기 $$x = 1$$에서 정답에 해당하는 함숫값은 $$y = 1$$이라고 알게 된 것이다. 그렇다면 우리는 $$ x = 1$$과 매우 가까운 다른 $$x$$, 예를 들면 $$x = 0.9, 1.1$$ 과 같은 점들에서 함숫값들이 여전히 0이라고 믿을 것인가? 아니면 우리가 새로 얻은 정답에 해당하는 $x$ 주변이므로 $$ y = 1$$에 가까운 y라고 생각할 것인가? GP의 핵심 개념은 이런 상황에서 후자를 택하여, 새로 얻은 점 주변에 값들은 그에 상응하는 공분산( 또는 커널 )을 활용하여 믿음을 업데이트 하는 것이다. 당연히 우리가 얻게 된 정답에 가까운 값들일수록, 새롭게 얻은 값에 의한 업데이트 효과가 클 것이고, 멀리 있는 값들일 수록, 새로 얻게 된 정답에 의해서 전혀 영향을 받지 않을 것이다. 우리가 현재 $$x = 1$$에서 $$y = 1$$이라는 힌트를 얻었는데, $$x = 100$$에서도 $$y$$가 1 주변일 것이라는 믿음은 어처구니 없지 않은가. 이렇게 $x$들 간의 거리에 기반하여 상관관계를 반영해 uncertainty를 줄여나가는 것이 GP의 핵심개념이라고 볼 수 있다. 이에 대한 얘기는 다음 섹션의 커널 관련 수식을 보면서 한 번 더 이해하도록 하자.

## d
<center><img src="https://imgur.com/ivDUi4n.png" width="50%" height="50%"><img src="https://imgur.com/ib8eTDe.png" width="50%" height="50%"></center>


> references
  https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
  http://www.gaussianprocess.org/gpml/chapters/
  http://cs229.stanford.edu/section/cs229-gaussian_processes.pdf
  https://en.wikipedia.org/wiki/Gaussian_process
  https://enginius.tistory.com/317
  https://math.stackexchange.com/questions/1176803/relationship-between-kriging-and-gaussian-process-regression-models?rq=1
