---
title: " [베이지안 추론] Binomial Proportion via Grid Approximation"
tags: Bayesian Statistics Inference Prior Posterior Likelihood Evidence Grid Approximation
---
\ 이전 챕터에서는 사전 분포가 베타 분포로 특정될 수 있을 때 어떻게 이항 비율에 대한 추론을 하는 지에 대하여 살펴 보았다. 직접적인 형식적 분석을 통해 쉽게 적분 계산을 해낼 수 있다는 점에서 베타 분포를 사용하는 것은 매우 편리하다. 하지만 만약에 베타 분포가 우리의 사전 믿음을 적절히 표현할 수 없다면 어떨까? 예를 들어, 우리의 믿음이 tri-modal이라고 상상해보자 : 우리의 동전이 tail로 매우 편향 돼있을 수도 있고, 거의 fair할 수도 있고, head쪽으로 매우 편향 돼있을 수도 있다. 어떠한 베타 분포도 이러한 세 개의 혹을 제대로 표현할 수 없다.<br>
\ 이 챕터에서, 촘촘한 grid들을 $\theta$값들에 걸쳐 표현하고 그에 대한 사전 분포를 정의함으로써 사후 분포에 대해 수치적으로 근사(approximate)해보고자 한다. 이 상황에서 우리는 $\theta$에 대한 사전 분포의 수학적 함수를 필요로 하지 않는다. 
