---
title: " [베이지안 추론] Binomial Proportion 추론"
tags: Bayesian Statistics Inference Prior Posterior Likelihood Evidence
---

# Inferring a Binomial Proportion via Exact Mathematical Analysis
이 파트에서 우리는 동전이 head가 나오는 경우의 확률을 측정해보고자 한다. 이 시나리오에서 우리에게 필요한 것은, 각각의 데이터가 보여주는 값들은 두 가지 경우의 수가 존재하며 해당 경우들이 서로 배타적인 확률적 공간을 보여주면 된다. 이 두가지 값들은 서로 관계가 없으며 단지 nominal 한 값들에 해당한다. 오로지 두가지의 nominal 값들이 있으므로 "binomial" 의 경우 또는 "dichotomous"한 경우로 부르자. 우리는 또한 각각의 데이터가 서로 독립적이며, 기저에 놓인 확률이 시간이 흐른다고 변하지 않는 stationary한 확률이라고 가정하자. 동전을 던지는 것은 이러한 상황의 대표적인 예시이다: 두 개의 가능한 결과 값이 있으며, 던지는 행위들은 서로 독립적이고, head를 얻을 확률은 시간이 지난다고 변하지 않는다. 이러한 이항 분포에는 다른 다양한 예시를 들어 볼 수 있으며, 생각보다 많은 현실 세계에서 이러한 이항분포의 예시가 많다는 것을 명심하자. <br>
베이지안 분석에서, 우리는 동전이 head를 보여 줄 가능한 확률들에 대한 몇몇 믿음에서 분석을 시작한다. 그리고, 우리는 베이즈 규칙을 사용해서 사후 분포를 추론한다. 베이즈 규칙은 우리로 하여금 likelihood함수를 특정하도록 요구하며, 이것은 다음 섹션의 주제가 될 것이다.
## The Likelihood function: Bernoulli distribution
동전을 던져서 head가 나오는 확률은 파라미터에 대한 함수로 볼 수 있다 : $$p(y=1|\theta) = f(\theta)$$. 우리는 특정한 간단한 함수로 이를 추정하며 identity 함수: $$p(y=1|\theta)=f(\theta)$$라고 하자. 이항분포 전제 조건(배타사건)에 따라, $$p(y=1|\theta)=\theta$$가 된다. 이 두 공식을 하나의 식으로 표현하면 다음과 같다:<br>
<center>$$p(y|\theta)={\theta}^y{(1-\theta)}^{(1-y)}\qquad(5.1)$$</center><br>
위의 공식은 Bernoulli distribution(베르누이 분포)로 불려진다. 베르누이 분포는 두 개의 discrete(이산)변수에 대한 확률 분포이며, 이때 $\theta$값은 고정 돼 있다고 생각한다. 특히, 확률들의 총 합은 1이며, 이에 대한 식은 다음과 같이 표현한다.<br>
<center>$$\sum_y p(y|\theta)=p(y=1|\theta)+p(y=0|\theta) = \theta + (1-\theta) = 1$$</center><br>
(5.1)식은 주어진 $\theta$ 값에서 datum $y$가 나올 확률을 계산하는 $\theta$에 대한 $likelihood\ function$으로 볼 수 있다. 베르누이 분포는 두개의 $y$ 값에 대한 이산 분포에 해당하지만, $likelihood\ function$은 연속 변수 $\theta$에 해당하는 함수임을 확인하자. likelihood function은 각각의 $\theta$값들에 대한 확률을 계산해주지만 이것은 확률 분포가 아님을 알아야한다. 만약 우리가 $$y=1$$이라고 가정한다면, $$\int_{0}^{1} d\theta\ {\theta}^y(1-\theta)^{(1-y)} = \int_{0}^{1} d\theta\ \theta =1/2\ne 1$$ 이기 때문에 확률 분포로 볼 수는 없다.<br>
베이지안 추론에서, 함수 $$p(y|\theta)$$는 불확실한 변수인 파라미터 값 $\theta$와, 고정되며 알려진 값 $y$의 데이터로 생각이 된다. 그러므로 $$p(y|\theta)$$는 일반적으로 $\theta$에 대한 likelihood function으로 부르며, 식 (5.1)은 $Bernoulli\ Likelihood\ function$ 으로 불려진다. <br>
 우리가 동전을 $N$번 던지고 그때 나오는 데이터의 셋을 $$D={y1,...,y_N}$$로 표현하게 되면, 그때 이 특정 집합 값을 얻게 될 확률은 각각 결과 확률들의 총 곱과 같을 것이다.:
<center>$$p({y1,...,y_N}|\theta) = \prod_{i} p(y_i|\theta)$$</center><br>
<center>$$ = \prod_{i}^{N} \theta^{y_i} (1-\theta)^{(1-y_i)}\qquad(5.2)$$</center><br>
만약 동전을 던지는 행위의 집합들에서 head가 나오는 횟수를 $$z=\sum_{i}^{N} y_i$$라고 한다면, (5.2)식은 다음과 같이 작성될 수 있다.<br>
<center>$$p(z,N|\theta)=\theta^z(1-\theta)^{(N-z)}\qquad(5.3)$$</center><br>
## A description of beliefs: The beta distribution
이번 챕터에서, 우리는 믿음에 대한 사후 분포의 수학적 형태를 유도하기 위해 순전히 수학적 분석만을 사용할 것이다. 수치적인 approximation은 사용하지 않는다. 이를 진행하기 위해서 우리의 사전 믿음에 대한 수학적인 묘사가 우선 필요하다. 즉, [0,1] 범위에 존재하는 $\theta$값들을 활용하여, 사전 믿음 확률을 계산하기 위한 수학적인 식이 필요하다.<br>
원칙적으로는, [0,1] 사이의 정의역을 취할 수 있는 어떠한 확률 밀도 함수를 사용해도 된다. 그러나 우리가 베이즈 룰을 적용하려고 할 때, 수학적 계산의 편의를 위해 두가지 신중히 고려해야 할 사안들이 있다. <br>
첫째, 베이즈 규칙의 분자에 해당하는 $$p(y|\theta)$$ 값과 $$p(\theta)$$의 곱이 $$p(\theta)$$와 같은 형태의 함수 결과를 보여주면, 편리할 것이다. 이러한 경우에, 사전 믿음과 사후 믿음은 모두 같은 형태의 함수를 사용해서 묘사가 된다. 이러한 특성을 사용하게 되면, 우리가 이후 추가적인 데이터를 포함시키고, 또 다른 사후 분포를 이끌어 낼 때, 또 다시 사전 믿음과 같은 형태로 이끌어내는 용이함을 준다. 그러므로, 얼마나 많은 데이터를 우리가 포함시키느냐에 상관없이, 우리는 항상 같은 함수적 형태를 띄는 사후분포를 보여줄 수 있다. <br>
둘째로, 우리는 베이즈 규칙의 분모에 해당하는 $$\int d\theta p(y|\theta)p(\theta)$$가 분석적으로 solvable하기를 희망할 것이다. 이러한 특성 또한 함수 $$p(\theta)$$의 형태가 얼마나 $$p(y|\theta)$$의 형태와 관련이 있는지에 영향을 받을 것이다. 만약 사후 분포가 사전 분포의 형태가 같은 모습을 가질 수 있도록, $$p(y|\theta)$$와 $$p(\theta)$$의 형태가 combine 돼 있다면, 이 때 $$p(\theta)$$는 $$p(y|\theta)$$에 대한 $$conjugate\  prior$$라고 불려진다. 사전 분포는 오직 특별한 형태의 likelihood function에 대해서만 conjugate하다는 것에 대하여 이해하고 넘어가자.<br>
현재 상황에서, 우리는 $\theta$에 대한 사전 밀도의 함수적 형태를 찾고 있으며, 이 형태는 식 (5.1)에 보여지는 베르누이 likelihood function과 conjugate 해야 한다. 만약 사전 분포의 형태가 $$\theta^a(1-\theta)^b$$의 형태를 갖고 있다면, 우리가 베르누이 likelihood를 사전분포와 곱해도 여전히 우리가 같은 형태의 함수, 즉 $$\theta^{(y+a)}(1-\theta)^{(1-y+b)}$$를 얻다는 것을 알 수 있다. 따라서 우리는 $$\theta^a(1-\theta)^b$$를 포함하는 확률 밀도 함수를 사용하고자 한다.<br>
이러한 형태의 확률 밀도는 $$beta\ distribution$$이라고 불려진다. 공식적으로 이 베타 분포는 $a$와 $b$ 두개의 파라미터를 갖고 있다. 확률 밀도는 다음과 같이 정의된다. <br>
<center>$$p(\theta|a,b)=beta(\theta;a,b)$$<br>$$=\theta^{(a-1)}(1-\theta)^{(b-1)}/B(a,b)\qquad(5.4)$$</center><br>
여기서 $B(a,b)$는 단순한 normalizing 상수이며 이것은 베타 분포 아래 부분의 면적이 1로 적분되도록 돕는 역할을 한다.<br>
베타 분포는 [0,1] 구간에 있는 $\theta$값들에 대해서만 정의가 된다는 것을 기억하자. $a$ 값과 $b$ 값은 양수이다. (5.4)식의 베타 분포의 정의에서 $\theta$ 값은 $$(a-1)$$ 만큼 제곱되고, $$(1-\theta)$$ 값은 $$(b-1)$$만큼 제곱된다는 것에 유의하자. 또한, 베타 함수, $$B(a,b)$$를 베타 분포, $$beta(\theta;a,b)$$와 혼동하지 않도록 하자. 베타 함수는 $\theta$에 대한 함수가 아니다. Figure 5.1의 각각의 panel들은 다양한 $$a,b$$값들에 대한 베타 분포를 보여준다.<br>
!['Imgur'](https://imgur.com/RJw088s.png)
## Specifying a beta prior
우리의 사전 믿음을 묘사하는 베타 분포에 대해 조금 더 구체화해보자. 베타 분포의 평균과 분산 값에 대해 아는 것은 유용할 것이다. $$beta(\theta;a,b)$$의 평균 값 $$\bar{\theta}\ =\ a/(a+b)$$ 이다. 그러므로 $$a\ =\ b$$일 때, 평균은 0.5이며, $a$값이 상대적으로 커지면 커질 수록 평균도 커지게 된다. 베타 분포의 표준 편차는 $$\sqrt{\bar{\theta}(1-\bar{\theta})/(a+b+1)}$$이다. 이 표준편차 값은 $$a\ +\ b$$ 값이 커지면 커질수록 작아지게 된다.<br>
사전 분포 안의 $a$와 $b$값은 각각 이전에 관측된 데이터들로 생각해볼 수 있다. 마치 $$a\ + \ b$$번의 동전 던지기에서 $a$번의 head와 $b$번의 tail이 나온 경우와 같다고 이해하면 된다.
