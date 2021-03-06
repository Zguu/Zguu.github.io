---
title: " [베이지안 추론] Metropolis Algorithm(2)"
tags: Bayesian
---
# Chapter 7. Inferring a Binomial Proportion via the Metropolis Algorithm

## The Metropolis algorithm more generally

## 7.2 The Metropolis algorithm more generally

우리가 앞에서 살펴본 metropolis 알고리즘의 예시는 일반적인 알고리즘의 매우 특별한 케이스를 살펴본 것이다. (현실에서 사용하기는 힘든, 엄격한 제약사항 하에서의 알고리즘 증명이다.) 이전 섹션에서 우리가 고려했던 것들은 다음과 같다.

* 연속적이지 않은 이산적인 포지션
* 1차원 움직임
* 오직 왼쪽 또는 오른쪽으로의 proposal

이 간단한 상황들은 우리가 알고리즘의 작동과 과정을 이해하는데 상대적으로 도움을 주었다. 하지만 이 알고리즘이 더욱 유용하려면 다음과 같은 반대의 상황에서 적용 가능해야한다.

* 연속적인 포지션(값들)
* 1개 이상의 다차원
* 더욱 일반적인 proposal 분포

 일반적 방법의 essential은 심플한 케이스에서와 같다. 첫째로, 우리는 어떤 타겟 분포 $P(\theta)$를 갖고 있다. 이것은 다차원의 연속 변수 공간에 존재하며 이 공간에서 우리는 샘플 값들의 대표 값들을 생성해내고자 한다. 우리는 어떠한 $\theta$들에 대해서도 $P(\theta)$값을 계산해낼 수 있어야 한다.(즉 우리는 $\theta$에 대한 분포를 알며, 이를 함수로 표현하고, 함수에 $\theta$ 값을 입력해서 함숫값을 얻어낼 수 있어야 한다.) 그러나 이 분포 $P(\theta)$가 normalized 돼 있을 필요까지는 없다. 단지, non-negative, 즉 양수 값을 갖는 분포이면 충분할 것이다. 특정한 적용 상황에서 $P(\theta)$는 $\theta$에 대한 normalized 되지 않은 사후 분포일 것이다.

타겟 분포로부터 얻은 샘플 값들은 파라미터 공간을 통해 random walk를 진행함으로써 생성될 수 있다. 이 walk는 임의의 한 점에서 시작되며 유저에 의해 단순히 정해진다. 이 스타팅 포인트는 어딘가 $P(\theta)$가 0이 되지 않는 점은 돼야 한다. Random walk는 매 번 파라미터 공간에서 새로운 포지션으로 이동하는 것은 제안하면서 진행된다. 그리고, 그 제안된 움직임을 받아들일지 아닐지를 결정하면서 진행된다. Proposal 분포는 많은 다른 형태를 가질 수 있고, $P(\theta)$가 가장 큰 질량을 갖는 파라미터 공간 지역을 효율적으로 찾아내도록 proposal 분포를 사용한다는 목적을 갖고 있다. 즉 (물론), 우리는 빠르게 랜덤한 값들을 생성해내기 위한 목적으로 proposal 분포를 사용해야 한다. 우리의 목적에 맞게, (목적을 위해) generic한 케이스를 고려해보자 (proposal 분포가 노말하고, 현재 포지션에 centered 돼있는). Normal 분포를 사용하는 데에 깔려있는 아이디어는, proposed 움직임이 현재의 포지션 주변에 있을 것이라는 것이고, 노말 커브에 의해 멀리 있는 포지션을 제안하는 확률은 drop off된다. R과 같은 컴퓨터 언어는 pseudo-랜덤한 값들을 노말 분포로부터 추출하는 내장 함수를 갖고 있다. 새로운 포지션으로의 제안을 만들어 낸 이후에 알고리즘을 해당 제안을 받아들일지 아닌지에 대해 결정한다.

## 7.2.1 “Burn-in”, efficiency, and convergence

만약 타겟 분포가 매우 퍼져있지만, proposal 분포는 매우 narrow하다면, random walk가 해당 분포를 모두 커버하는 데에는 긴 시간이 걸릴 것이다. 수천 킬로미터 너비로 매우 넓은 전체 대륙에서 우리가 서베이를 진행하려고 할때, 동전을 하늘에 던져가며 앞면이 나오는지 뒷면이 나오는 지에 따라 수미터씩 움직이면 얼마나 비효율 적인 서베이 방식일지 상상해보라. 그러므로, proposal 분포가 매우 narrow 할 때, Metropolis 알고리즘은 매우 효율적이지 않다. 전체 대표 값 샘플들을 누적하기 위해 너무 많은 step이 필요하다. 만약 우리가 타겟 분포 상의 매우 평평하고 낮은 지역에서 우연치 않게 initial point를 갖게 된다면, 실제로 매우 dense한 타겟 분포의 지역으로 포지션이 움직이는 데에 매우 많은 시간이 걸릴 것이다.이러한 문제를 경감시키기 위해서, random walk의 초기 스텝은 타겟 분포의 대표 값들에서 간주되지 않는다 (배제된다).
이렇게 초기 스텝들을 우리가 배제시키는 구간은 “burn in” 구간으로 불려진다.


랜덤 워크가 오랫동안 배회하는 동안(meandered), 우리는 이 exploring 과정이 실제로 타겟 분포를 정말 탐험하고 있는 지에 대해 확신할 수 없다. 특히 만약 타겟 분포가 차원 수가 높은 매우 복잡한 상황이라면 더욱 그럴 것이다. Random walk의 수렴에 대해 평가하려고 하는 다양한 방법론들이 있다. 하지만 이 챕처에서 우리는 해당 탐험에 대한 뉘앙스(더 깊게 파고드는 미묘한 의미를 뜻하는 듯)까지 다루지는 않는다. Application 자체는 매우 심플하기 때문이다.
너무 협소한 제안 분포의 문제에 대해서 지적했던 이전의 고려는, 그렇다면, 이 제안 분포를 매우 넓게 잡는 것이 최선이 아니겠냐는 생각으로 우리를 이끌 수도 있다. 하지만 너무 과한 너비 또한 문제를 낳을 수 있다. 만약 제안 분포가 너무 넓게 되면, 우리의 $$P(\theta_{current})$$가 너무 높은 경우에, 분포의 중앙에서 너무 멀리 떨어진, 즉 너무 낮은 $$P(\theta_{proposed})$$, 에서 비율로 제안의 accept가 결정되게 된다. 결과적으로 이동에 대한 제안은 거의 받아들여지지 않고, random walk는 해당 포지션에서 멈추고 머물게 된다. 결과적으로, 적당한 분산으로 잘 조정된 제안 분포를 선택하는 것, burn-in 스텝의 양을 잘 조정하는 것, random walk의 수렴성에 대해 잘 조정하는 것 모두 매우 쉽지 않은 과정이다. 그러나 이번 챕터에서 타겟 분포는 이러한 문제들이 큰 문제가 되지 않을 정도로 well-behaved 된다. 그럼에도 불구하고, 7.1 exercise에서 우리는 Metropolis 알고리즘의 한계에 대해 깨닫는다.

## 7.2.2 Terminology: Markov chain Monte Carlo

대표 랜덤 값들을 생성해냄으로써 타겟 분포의 특성을 평가하는 것은 Monte Carlo 시뮬레이션의 케이스이다. 특정 분포로 부터 많은 랜덤 값들을 샘플링 하는 이러한 시뮬레이션은 Monte Carlo 시뮬레이션이라고 불려진다. Metropolis 알고리즘은 Monte Carlo 프로세스의 특정한 타입이다. 이 프로세스는 이전 포지션과는 매우 독립적으로 각각의 스텝을 만들어 낸다. 제안된 다음 스텝은 이전에 어떠한 스텝이 이뤄졌는 지에 대해서 전혀 독립적이지 않고, 해당 제안 스텝을 받아들일지 거절할지 또한 이전 스텝과는 전혀 관련이 없다. 이렇게 이전의 상태에 대한 어떠한 memory가 없는 이러한 스텝이 (first order) Markov process로 불려진다. 그리고 이러한 스텝의 연속을 Markov chain이라고 부른다. Metropolis 알고리즘은 Markov Chain Monte Carlo (MCMC) 프로세스이다.

## 7.3. From the sampled posterior to the three goals

우리가 지금까지 MCMC라는 나무들에 집중해 왔다면, 이제는 Bayesian 추론이라는 숲의 관점에서 더욱 집중을 해보자. Bayesian 추론에서, 우리는 posterior 분포에 대한 좋은 묘사를 필요로 한다. 만약 우리가 formal한 분석을 통해서든, dense-grid approximation 을 통해서든 어떻게도 해당 분포에 대한 묘사를 잘 할 수 없다면, 우리는 posterior 분포로부터의 많은 대푯값들을 샘플링하고 이 값들을 이용해 posterior에 대한 대략적 추정을 할 수 있다. 지금까지 이 챕터에서 우리는 어떤 분포로부터 특정 대푯값들을 생성해내는 한 과정을 살펴봤다 (Metropolis algorithm). 우리는 이 책에서 동전이 앞면을 바라보는 확률들을 추정하는 간단한 케이스에 대해 집중하고 있다. 다르게 말하면, 우리는 underlying 확률 $\theta$에 관한 posterior 믿음에 대해 추정하려고 하고 있다. 사전 믿음 분포인 $$p(\theta)$$로 시작 해보자. 현재 시나리오에서 $$p(\theta)$$는 수학적으로 $\theta$에 대한 함수의 형태를 취하고 있다. 수학적 함수 $$p(\theta)$$의 특정 값은 어떠한 값의 $\theta$에 대해서도 쉽게 계산이 가능해야 하며, 이에 대한 좋은 예시가 베타 분포 형태이다.
 우리는 또한 수학적 우도 함수 $$p(D|\theta)$$ 또한 갖고 있다. 한 번의 동전을 던지는 시행에서 우도 함수는 베타 분포를 따른다. $$p(y|\theta) = \theta^y (1-\theta) ^ {(1-y)}$$. 여러번의 독립적인 동전을 던지는 시행에 있어서 likelihood는 각각의 던지는 시행에 있어서 확률들의 곱이 된다.
 사후 분포 $$p(\theta|D)$$ 는 베이즈 법칙에 따라, $$p(D|\theta)p(\theta)$$ 에 비례한다.

 우리는 이 곱을 Metropolis 알고리즘의 타겟 분포로 사용한다.Metropolis 알고리즘은 절대적인 사후 분포 확률을 필요로 하는 것이 아니라, 상대적인(또는 비례하는) 타겟 분포의 사후 확률을 필요로 한다. 따라서 우리는 unnormalized prior and/or unnormalized posterior를 theta의 샘플 값을 생성하는 데에 사용할 수 있다. 나중에 우리가 모델 간의 비교를 하는 목적에 있어서는, $p(D)$를 추정할 필요가 있으며 이 상황에서는 우리는 실제 사후 확률의 normalized 값을 필요로 한다. 우리는 몇몇 후보 theta 값들에서 Metropolis 알고리즘의 random walk를 시작한다 ($\theta$ = 0.5 같은). 그리고 새로운 포지션으로 점프를 제안한다. 이러한 제안 분포는 정규분포 일수도 있다 (std = 0.2라고 하자). 이러한 std 선택은 왜 합리적일까? 우선, theta값은 [0,1] 사이에 존재하며, 확실이 우리는 theta값의 범위에 비해 제안 분포가 협소하길 원할 것이다. 또다른 고려 사항은, 제안 분포가 사후 분포의 넓이에 조정이 돼야 한다는 것이다. 너무 넓지도, 너무 좁지도 않게. 샘플의 사이즈가 작은 경우에, 사후 확률은 전형적으로 매우 좁으며, std = 0.2 정도가 적절할 수 있다. 그러나 제안 분포와, 이에 대한 표준편차와 같은 분포의 특성들은 우리와 같은 분석가들이 직접 정해야 한다는 것을 잊어선 안된다. exercise 7.1은 다른 제안 분포들의 각기 다른 결과에 대해 보여준다. 제안 분포로써 정규 분포가 사용되면, 제안되는 값은 0보다 작을 수도 있고, 1보다 큰 경우가 생기는데 이러한 $\theta$ 값들은 적절하지 않다. 만약에 이렇게 부적절한 $\theta$ 값이 제안 되는 경우에 사전분포와 가능도 함수가 0을 리턴한다면 상황은 괜찮을 것이다. Section 7.6.1 에서의 R code는 Metropolis 알고리즘에 대한 샘플을 보여준다.
!['Img'](https://imgur.com/Tofepwf.png)
 해당 코드에 대해서 (굳이) 자세히 살펴보자면, 이 코드는 세가지 함수에 의해 정의되는 것을 볼 수 있다. : The Likelihood 함수, 사전 확률 함수, 목표 분포 함수이다.<br>
 $\ $일반적으로, 목표 분포 함수는 간단히 가능도와 사전확률의 곱이다. 타겟 분포를 정의한 이후에, 그 다음 코드들은 random walk들을 생성해 낸다. 파라미터 공간에서 잠시 주변을 배회한 이후에, 코드는 burn-in 부분을 임의로 제외하며, 나머지 walks들만을 사후 분포의 대표값들로 인지하고 저장한다.<br>
 $\ $Metropolis 알고리즘은 Figure 7.3에 나타나 있다. $\theta$값들은 uniform 사전분포, 베르누이 가능도, 그리고 $$z\ =\ 11$$ and $$N\ =\ 14$$라는 데이터로부터 생성된다. 이렇게 생성된 Figure 7.4을 Figure 5.2(수학적 분석으로 생성된) 및 Figure 6.2(Grid 근사로 생성된) 결과값들과 비교해보자. 매우 비슷하다.
## 7.3.1 Estimation
$\ $random walks를 통해 생성된 $p(\theta|D)$의 대푯값들로부터, 우리는 실제 $p(\theta|D)$의 분포에 대해 평가할 수 있다. 예를 들어, 이 대푯값들의 central tendency를 요약하기 위해, 간단히 평균 또는 중앙값을 이용할 수 있다. 이런 분포의 평균 값은 Figure 7.3에 잘 나타나 있다.
### 7.3.1.1 Highest density intervals from random samples
$\ $Highest density intervals(HDIs)는 MCMC 샘플들로부터 estimate될 수 있다. 이를 이용하는 한가지 방법은, 샘플의 각각의 포인트들에서 상대적인 사후 확률을 계산하는데에 의존하는 것이다. 기본적으로, 이 방법은 많은 점들 중에서 95% HDI 구간 바깥에 있는 5%의 점들이 수면 아래에 있도록 한다는 개념이다. 조금 더 구체적으로 살펴보면, 첫 번째로 우리는 각각 점들에서 상대적인 높이를 계산한다. 이는 상대적인 사후 밀도에 해당하며 각 $\theta_i$ 값들에서 $$p({\theta}_i|D)\ \propto\ p(D|{\theta}_i)p({\theta}_i)이다. $$p_{.05}$$를 모든 사후 높이 값들의 5th percentile이라고 해보자. 이것은 95% HDI 끝 가장 자리에 있는 높이와 같다고 볼 수 있다. waterline 위에 있는 모든 점들은 HDI를 나타낸다.
### 7.3.1.2 Using a sample to estimate an integral
$\ $해당 분포로부터 우리가 꽤나 많은 대표값들을 갖게 됐다고 가정하자. 해당 분포의 평균에 대한 좋은 추정을 무엇일까? 직관적인 대답은 다음과 같다: 샘플들의 평균이 원래 분포의 평균이라는 것이다. 왜냐하면, 이미 우리는 해당 분포를 충분히 나타낼 풍부한 대표값들을 많이 갖고 있기 때문이다. 다르게 얘기하면, 분포로부터 샘플링된 이산 값들에 대한 합을 진행함으로써 (결국 적분의 근사인), 평균을 근사할 수 있다는 것이고, 이에 대한 논의는 이미 앞의 단원에서 많이 선행한 것이다. 이 근사 방법을 공식으로 표현할 수 있다. $$p(\theta)$$를 파라미터 $\theta$에 대한 분포라고 하자. $\theta_i$는 분포 $$p(\theta)$$로부터 샘플링된 값이라고 하자. 이런 경우에 우리는 $$\theta_i ~ p(\theta)$$로 표현하며, 이는 $\theta_i$ 값들이 확률 분포 $$p(\theta)$$에 의해 샘플링됐다는 것을 의미한다. 샘플의 평균으로부터 적분을 통해 근사된 실제 평균은 다음과 같다.<br>
!['Img2'](https://imgur.com/EYJLMbz.png)
> 수식 (7.5)의 왼쪽 항에는 $$p(\theta)$$가 존재하는 데에 반해, 오른쪽 항에는 존재하지 않는다. 오른쪽 항에는 $\theta$ 가 아닌 ${\theta}_i$ 로 존재하며, 이 값은 $p(\theta)$ 분포에서 추출된 것이므로, 추출 과정에서 이미 $p(\theta)$ 의 확률 밀도 값을 반영한다고 생각한다.

수식 (7.5)에서 보여지는 이 근사 값은 $N$이 커지면 커질수록 더 정확해 질 것이다. 수식 (7.5)는 일반적 원칙의 특별한 케이스일 뿐이다. 모든 함수 $f(\theta)$에 대하여, 확률 분포 $$p(\theta)$$에 의해 weighted된 해당 함수의 적분은 샘플링된 포인트 들에서의 함숫값의 평균으로 근사될 수 있으며 수학적으로는 아래와 같이 표현된다.<br>
!['Img3'](https://imgur.com/202E55E.png)
수식 (7.5)에서의 평균의 근사값은 $$f(\theta) = \theta$$이다. 수식 (7.6)은 이 챕터의 나머지 부분들에서 주로 다룰 내용이다. 이 식에 대하여 우리 스스로 조금 더 깊게 이해하고 넘어가야 할 필요가 있다. <br>
$\ $ 수식 (7.6)의 왼쪽 항에 있는 적분을 이산화하는 것에 대하여 생각해보자. 아주 많은 작은 구간들에 걸쳐서 근사를 진행한다: $$\int d\theta f(\theta) p(\theta) \approx \sum_{j} [\Delta \theta p({\theta}_j)] f({\theta}_j)$$ 이며, ${\theta}_j$는 $j$번째에 있는 $\theta$ 값을 나타낸다.
> 간단한 구분 구적법을 진행한다.

대괄호 안에 있는 항인 $$\Delta \theta p({\theta}_j)$$는 ${\theta}_j$ 주변에서의 작은 구간의 확률 질량으로 볼 수 있다. 전체 샘플링 된 수를 $N$으로 표현하고, 해당 구간 $$j^{th}$$에서 $\theta$값을 얻는 횟수를 $$n_j$$로 표현하자. 샘플이 많아지면 많아질 수록,
$${n}_j/N \approx \Delta \theta p({\theta}_j)$$가 성립하게 된다. 그래서 다음과 같은 식이 성립하게 된다: $$\int d\theta f(\theta) p(\theta) \approx \sum_{j}[\Delta \theta p({\theta}_j)] f({\theta}_j) \approx \sum_{j} [n_j / N] f({\theta}_j) = 1/N \sum_j n_j f({\theta}_j)$$. 다르게 얘기하면, 우리가 $j_{th}$ 구간에서 $\theta$ 값을 샘플링 할 때마다, another iteration of the interval's representative value,$$f({\theta})$$를 summation에 추가해준다. 하지만, 구간의 대표값을 사용할 필요는 없다; 단지 샘플링된 $\theta$ 값의 함수 값인 $$f(\theta)$$만을 사용한다. 샘플링된 ${\theta}_j$는 이미 $j^{th}$ 구간에 존재하기 때문이다. 따라서 근사식은 다음과 같이 되며 이는 수식 (7.6)과 같다. $$\int d\theta f(\theta)p(\theta) \approx 1/N \sum_{j} n_j f({\theta}_j) \approx 1/N \sum_{\theta}_i \approx p(\theta)}^N f({\theta}_i).
### 7.3.2 Prediction
$\ $베이지안 추론의 두 번째 분석 목표는 subsequent 데이터 값들에 대한 예측을 하는 것이다. $$y\ \in\ {0, 1}$$ 의 데이터에 대하여, $y$의 에측될 확률은 $$p(y|D) = \int d\theta p(y|\theta) p(\theta|D)$$이다. 이 식이 수식 (7.6)의 왼쪽 항과 같은 형태를 취하고 있음을 숙지하자. 따라서, 해당 식을 $y$가 1과 같을 예측 확률에 적용해보면, 우리는 다음을 얻는다.<br>
!['Img4'](https://imgur.com/PevvXyX.png)
>계속해서 비슷한 형태의 수식들이 계속해서 나오고 있지만, 미세하게 다름을 인지해야 한다. 수식 (7.5)와 비교해보자.

### 7.3.3 Model comparison: Esitmation of $p(D)$
$\ $ 이번 섹션은 우선 스킵해둔다.


## 7.4 MCMC in BUGS
$\ $섹션 7.2.1로 돌아가보면, Metropolis 알고리즘은 제약 사항이 꽤 많았었다: proposal 분포가 사후 분포와 잘 맞아떨어져야 했고, 초기 구간에서 샘플들은 burn-in 됐다. 또한, 샘플링 chain이 충분히 오랫동안 작동해야 했다. R에서 이런 제약 사항들이 자동으로 고려되어 작동되는 패키지는 이미 존재한다. BUGS (Bayesian inference Using Gibbs sampling). 나머지 챕터에서 우리는 Gibbs sampling에 대해 조금 더 깊게 알아보도록 하자.<br>
$\ $Gibbs sampling은 Metropolis 알고리즘의 일종이다. 우리는 BUGS의 OpenBUGS 버전을 예를 들어 사용하도록 한다. 해당 패키지는 베이지안 모델을 사용하며 MCMC 사후 샘플들을 생성해낸다.
