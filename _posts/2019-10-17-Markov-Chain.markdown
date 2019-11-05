---
title: " [베이지안 통계학]사전확률? 사후확률? "
tags: Statistics
---


# 0. Prior(사전확률)과 Posterior(사후확률)
   베이즈 규칙에서 Prior(사전확률)과 Posterior(사후확률)에 대한 이해를 돕기 위해 간단한 동전의 경우를 살펴보자. 흠집이 없이 아주 대칭의 모양을 지닌('fair한') 동전은 그 동전을 던졌을 때, 50%의 확률로 앞면이, 50%의 확률로 뒷면이 나오게 될 것이다. 하지만 실제 경우에서 동전의 변형 또는 다른 다양한 이유로, 우리는 이 50%, 50% 확률이 때때로 왜곡된다고 생각할 수도 있다.
 예를 들어, 동전 모양이 꽤나 왜곡돼서, 우리는 현재 이 동전을 던지면 20%의 확률로 앞면이, 80%의 확률로 뒷면이 나올 것이라고 믿고 있다. 이렇게 우리가 실제로 동전을 던져보기전에 갖고 있는 사건(event)에 대한 믿음을 사전확률이라고 하자.
 <br>
 만약, 실제로 동전을 10번 던진 후 관찰 결과, 앞면이 2번, 뒷면이 8번 나왔다면, 우리는 우리가 이 사건 전에 지니고 있던 확률, 즉, Prior(사전확률)에 대한 믿음을 더욱 확고히 하게 된다.
 <br>
 우리가 학교나 책을 통해 이 Prior와 Posterior에 대한 개념을 처음 접할 때, 보통 시간 순서로 (이전에 일어난 확률) 과 (이후에 일어난 확률)로 받아들인다. 하지만 이렇게 이해하는 것보다는, 특정한 데이터 셋(우리가 앞에서 본 동전을 던져 실제로 나온 비율의 값들)을 배제했을 때 우리의 믿음을 Prior, 포함시켰을 때 우리의 믿음을 Posterior로 이해하는 것이 낫다.
<br>
데이터를 활용한 추론의 목표에는 크게 세 가지 종류가 있다.
- parameter values Estimation<br>
- Prediction of data values<br>
- Model comparison<br>

## Bayes' Rule accounting for Data

$$p(\theta|D) = posterior , 사후확률$$<br>
$$p(D|\theta) = likelihood , 가능도$$<br>
$$p(\theta) = prior , 사전확률$$<br>
$$p(D) = evidence , 관측확률$$<br>
사전확률 = 데이터가 주어지지 않았을 때, theta 값의 분포 함수<br>
$$ where the evidence is p(D) = \int d\theta p(D|\theta)p()\theta) $$<br>
Bayes' Rule : $$p(\theta|D) = p(D|\theta)p(\theta)/p(D)$$

## An example with coin flipping

동전을 던지는 예시에서 사전확률과 가능도, 사후확률의 분포에 대하여 아래 그래프를 보며 이해해보자. 우리는 현재 이 동전이 fair하다고 생각하므로, 우리의 사전 확률 분포는 아래의 맨 위 그래프 분포와 같다. 해당 사전분포를 가진 상태에서, 앞면이 3회, 뒷면이 9회 나올 확률을 계산해보면 해당 식은 $$p(D|\theta) = {\theta}^3(1-\theta)^9** 의 형태를 띄게 된다. 따라서 사전 확률 분포에서 \theta 값이 0.25일 때 가능도는 가장 높고, 0.5일 때가 그 뒤를 따르며, 0.75일 때 가능도는 사실상 0에 가깝게 된다. 이렇게 가능도를 최대화시키는 \theta 값을 $$maximal likelihood estimate of \theta$$라고 부르자.<br>
이후에, evidence에 해당하는 $p(D)$에 대한 값을 계산해보자. 해당 값은

!['Imgur'](https://imgur.com/Kmd35Rn.png)
