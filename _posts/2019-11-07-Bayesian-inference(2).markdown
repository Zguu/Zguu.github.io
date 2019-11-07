---
title: " [베이지안 통계학] "
tags: Bayesian Statistics Inference Prior Posterior Likelihood Evidence
---

# Inferring a Binomial Proportion via Exact Mathematical Analysis
이 파트에서 우리는 동전이 head가 나오는 경우의 확률을 측정해보고자 한다. 이 시나리오에서 우리에게 필요한 것은, 각각의 데이터가 두 가지 경우의 수가 존재하며 해당 경우들이 서로 배타적인, 그러한 확률적 공간을 보여주면 된다. 이 두가지 값들은 서로 관계가 없으며 단지 nominal 한 값들에 해당한다. 오로지 두가지의 nominal 값들이 있으므로 "binomial" 의 종류 또는 "dichotomous"한 경우로 부르자. 우리는 또한 각각의 데이터가 서로 독립적이며, 기저에 놓인 확률이 시간이 흐른다고 변하지 않는 stationary한 확률이라고 가정하자. 동전을 던지는 것은 이러한 상황의 대표적인 예시이다 : 두 개의 가능한 결과 값이 있으며, 던지는 행위들은 서로 독립적이고, head를 얻을 확률은 시간이 지난다고 변하지 않는다. 다른 다양한 예시를 들어 볼 수 있으며, 생각보다 많은 현실 세계에서 이러한 이항분포의 예시가 많다는 것을 명심하자. <br>
베이지안 분석에서, 우리는 동전이 head를 보여 줄 가능한 확률들에 대한 몇몇 믿음에서 분석을 시작한다. 그리고, 우리는 베이즈 규칙을 사용해서 사후 분포를 추론한다. 베이즈 규칙은 우리로 하여금 likelihood함수를 특정하도록 요구하며, 이것은 다음 섹션의 주제가 될 것이다.
## The Likelihood function: Bernoulli distribution
