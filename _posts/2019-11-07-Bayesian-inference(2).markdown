---
title: " [베이지안 통계학] "
tags: Bayesian Statistics Inference Prior Posterior Likelihood Evidence
---

# Inferring a Binomial Proportion via Exact Mathematical Analysis
이 파트에서 우리는 동전이 head가 나오는 경우의 확률을 측정해보고자 한다. 이 시나리오에서 우리에게 필요한 것은, 각각의 데이터가 두 가지 경우의 수가 존재하며 해당 경우들이 서로 배타적인, 그러한 확률적 공간을 보여주면 된다. 이 두가지 값들은 서로 관계가 없으며 단지 nominal 한 값들에 해당한다. 오로지 두가지의 nominal 값들이 있으므로 "binomial" 의 경우 또는 "dichotomous"한 경우로 부르자. 우리는 또한 각각의 데이터가 서로 독립적이며, 기저에 놓인 확률이 시간이 흐른다고 변하지 않는 stationary한 확률이라고 가정하자. 동전을 던지는 것은 이러한 상황의 대표적인 예시이다 : 두 개의 가능한 결과 값이 있으며, 던지는 행위들은 서로 독립적이고, head를 얻을 확률은 시간이 지난다고 변하지 않는다. 이러한 이항 분포에는 다른 다양한 예시를 들어 볼 수 있으며, 생각보다 많은 현실 세계에서 이러한 이항분포의 예시가 많다는 것을 명심하자. <br>
베이지안 분석에서, 우리는 동전이 head를 보여 줄 가능한 확률들에 대한 몇몇 믿음에서 분석을 시작한다. 그리고, 우리는 베이즈 규칙을 사용해서 사후 분포를 추론한다. 베이즈 규칙은 우리로 하여금 likelihood함수를 특정하도록 요구하며, 이것은 다음 섹션의 주제가 될 것이다.
## The Likelihood function: Bernoulli distribution
동전을 던져서 head가 나오는 확률은 파라미터에 대한 함수로 볼 수 있다 : $$p(y=1|\theta) = f(\theta)$$. 우리는 특정한 간단한 함수로 이를 추정하며 identity 함수: $$p(y=1|\theta)=f(\theta)$$라고 하자. 이항분포 전제 조건(배타사건)에 따라, $$p(y=1|\theta)=\theta$$가 된다. 이 두 공식을 하나의 식으로 표현하면 다음과 같다:<br>
<center>$$p(y|\theta)={\theta}^y{(1-\theta)}^{(1-y)}$$</center><br>
위의 공식은 Bernoulli distribution(베르누이 분포)로 불려진다. 베르누이 분포는 두 개의 discrete(이산)변수에 대한 확률 분포이며, 이때 $\theta$값은 고정 돼 있다고 생각한다. 특히, 확률들의 총 합은 1이며, 이에 대한 식은 다음과 같이 표현한다.<br>
<center>$$sum_y p(y|\theta)=p(y=1|\theta)+p(y=0|\theta) = \theta + (1-\theta) = 1$$.</center><br>
