---
title: " [추천시스템] BPR : Bayesian Personalized Ranking from Implicit Feedback"
tags: RecommenderSystem
---

# Personalization & Ranking
FM에 이은 RENDEL 교수님의 또 다른 논문 BPR을 살펴보겠습니다.<br>
추천시스템에서 많이 사용되는 MF나 kNN 모두 유저 단위에서 추천을 진행할 수는 있습니다. (즉, 개인화를 추천에 포함시킵니다.)<br>
MF(Matrix Factorization)의 경우, 방법론 이름 그대로 행렬을 분해하는 과정에서, 유저들 간의 관계를 latent factor가 내포할 수 있습니다. 또한, 분해 결과 자체에서도 $U ^{|U| \times |k|}$ 형태의 $\text{user by factors}$ 행렬을 얻기 때문에, 각 유저의 특성에 맞는 추천을 진행한다는 것을 알 수 있습니다.<br>

또한, `kNN`의 경우, user vector 간의 가장 유사한 벡터를 찾아내느냐, item vector 간의 유사한 벡터를 찾아내느냐 크게 두가지 방법이 있습니다. 역시,가장 유사한 user vector를 찾는 과정을 사용할 시에, 나와 비슷한 유저를 찾아내서 추천을 진행하는, 개인화의 방법이 사용된다는 것을 알 수 있습니다.<br>

그렇다면 위의 두 기존 방법론 모두 `개인화 (Personalization)` 라는 개념을 동일하게 사용되는데, BPR 만의 특별한 점은 무엇일까 생각해볼 수 있습니다. BPR 의 마지막 키워드가 말해주고 있는 `Ranking` 에 바로 그 핵심이 있습니다.

> 실제 추천시스템을 구현하고 사용할 때, 단 한개만의 추천 결과를 사용하는 경우는 매우 드뭅니다. 보통은 여러 개의 상품들이나 컨텐츠를 동시에 유저에게 제공하기 때문에, 추천 결과들 간에 무엇이 더 효과적인지 순서를 매겨야합니다. 따라서, Ranking 을 학습하는 것이 매우 중요합니다.

## Ranking in MF & kNN

사실, MF 와 kNN 모두 추천 결과에서 추천된 리스트들 간의 순위를 매겨볼 수는 있습니다. MF의 결과로 나온 값들 중에 높은 값들로 상품들을 정렬하거나, 마찬가지로 kNN도 가장 유사도 점수가 높은 순서로 상품을 정렬하면 됩니다.

하지만, 두 알고리즘 모두 학습하는 과정에서 모델의 Loss 를 최소화하는 데에만 중점을 둘 뿐, 결과로 나온 추천 결과들이 제대로 정렬되는 방향인지에 대해서는 사실 크게 신경을 쓰지 않습니다.

동일한 추천시스템에서 나온 같은 추천 결과를 노출한다고 하더라도, 이 추천결과를 어떤 순서로 노출하냐에 따라서도 추천 시스템의 퍼포먼스는 크게 달라질 수 있습니다. 따라서, Ranking 을 학습하는 개념을 모델에 포함시키면 좋을 것입니다.

## Formalization

이 논문에서는 아이템 간의 선호도 대소관계 비교가 주를 이룹니다. 이해를 위해 간단한 부등식에 관련해서 설명합니다.

$U$는 모든 유저들의 집합을, $I$는 모든 상품의 집합을 의미합니다. 이를 매트릭스 형태로 나타내면 다음과 같은 행렬을 얻습니다. 이 행렬의 각 cell은 explicit feedback이 아닌, `implicit feedback` 값들을 지니고 있습니다. (1 ~ 5 와 같은 범위 값이 아닌, 0 또는 1의 값)

$$S \subseteq U \times I$$

우리가 만들고자 하는 추천 시스템은, 유저 $u$에게 모든 상품들을 어떤 순서로 제공할지 정해야 합니다. 즉, 상품들간에 줄을 세워야 합니다. 이러한 모든 상품들 간의 선호도 대소관계는 $>_u$ 로 표현하며 이 대소관계는 아래의 세 성질을 만족합니다.

- $\forall i,j \in I : i \ne j \Rightarrow i >_u j \vee j >_u i \ (totality)$

- $\forall i,j \in I : i >_u j \wedge j >_u i \Rightarrow i = j \ (antisymmetry)$

- $\forall i,j,k \in I : i >_u j \wedge j >_u k \Rightarrow i >_u k \ (transitivity)$

totailty 는, $i$와 $j$의 두 상품 중 더 큰 대소관계가 존재한다는 것을 의미합니다.<br>
antisymmetry 는, $i$와 $j$의 동등조건을 말하고 있습니다.<br>
transitivity는 $i,j,k$ 간의 삼단논법이 성립함을 의미합니다.

편의를 위하여 우리는 다음과 같은 집합도 정의합니다.

$$I_u^+ := {i \in I : (u,i) \in S} \\ U_i^+ := {u \in U : (u,i) \in S}$$

$I_u^+$는 유저 $u$가 평가를 남긴 상품들에 대한 pair 집합을 의미합니다. 마찬가지로, $U_i^+$는 상품 $i$에 대하여, 평가를 남겨준 유저들과의 pair 집합을 의미합니다.

> Formalization에서 나오는 이야기들은 사실 당연한 이야기들이며, 각 집합과 부등식들의 표현에 익숙해지며 이해하고 넘어가면 충분합니다.

## Problem Setting

유저들이 영화나 상품에 대한 평점을 1~5와 같이 명시적으로 남기는 경우에 이러한 점수 데이터들은 `explicit feedback` 이라고 합니다. 이와는 대조적으로 유저들이 해당 상품에 대한 선호도를 명시적으로 표현하지 않고, 단지 클릭만 하거나 구매를 하는 것과 같이 행동의 유무로 남긴 데이터들은 `implicit feedback` 이라고 합니다.

`implicit feedback` 데이터는 `explicit feedback` 데이터보다 그 데이터의 양은 풍부하지만, 사용하는 데에 있어서 분명 한계점이 존재하며 이를 인지하고 있어야합니다. 유저가 해당 상품을 구매하거나 클릭하여, `implicit feedback` 데이터를 남겼다고 하더라도, 우리는 이 데이터를 보고 해당 유저가 그 상품을 확실히 좋아한다 라고 결론을 내리기는 어렵습니다.

하지만, `implicit feedback`을 남긴 상품과 `implicit feedback` 조차 안남긴 두 개의 상품에 대한 선호도를 `비교` 해보라고 한다면, 아무래도 전자의 경우에 유저의 관심은 더 높지 않을까 라고 추측해 볼 수 있습니다.

이에 대한 저자의 행렬 표현방식이 아래 `Figure1` 과 `Figure2`에 표현 돼있습니다.

![스크린샷 2020-09-15 오후 1.14.05](https://i.imgur.com/q4XC6ZV.png)

![스크린샷 2020-09-15 오후 1.14.14](https://i.imgur.com/i2QjbJt.png)

위의 Figure들을 각각 예를 들어서 이해해보면, `Figure1` 에서 1번 유저 $u_1$은 상품 2,3번 $u_2, u_3$에 대해서는 `implicit feedback`을 남겼지만, 1,4번 상품에 대해서는 그렇지 않았습니다. 따라서 우리는 1번 유저가 2번 상품을 상대적으로 1,4번 상품보다는 선호한다고 표현할 것입니다. 이것이 `Figure2`의 상단 행렬에 표현 돼있습니다. (행렬의 1행 2열 = +, 행렬의 4행 2열 = +)

이와 같이 `implicit feedback` 들을 활용하여 상대적인 상품들 간의 선호도 대소관계를 표현하고, 이러한 모든 순서쌍들은 아래와 같이 표현할 수 있습니다.

$$D_S := \left\{ (u,i,j)|i\in I_u^+ \wedge j \in I \backslash I_u^+ \right\}$$

## BPR-OPT & LearnBPR

이제 위와 같은 problem setting 상황에서, 개인화된 랭킹 문제를 해결하는 일반적인 방법론을 도출해보겠습니다.

첫번째로, 개인화된 랭킹 문제를 해결하는 최적화 기준인 BPRopt는 likelihood 에 해당하는 $p(i >_u j | \Theta)$ 와, parameter의 사전확률에 해당하는 $p(\Theta)$ 에 대한 Bayesian 분석을 통해 도출됩니다.

**BPR Optimization Criterion**

우리의 목적은 아래의 posterior 확률을 최대화하는 것입니다. 이 posterior 함수는, 상품들 간의 대소관계들이 주어졌을 때, parameter에 해당하는 $\Theta$ 의 확률이며, 이는 앞서 언급했던 likelihhood 와 prior 확률의 곱에 비례합니다.

$$p(\Theta|>_u) \propto p(>_u|\Theta)p(\Theta)$$

각각의 유저들에 대한 확률들과, 각 상품의 대소관계는 다른 상품들의 대소관계와 독립적이라고 가정하므로, 아래와 같이 다시 표현할 수 있습니다.

$$\prod_{u\in U} p(>_u | \Theta) = \prod_{(u,i,j)\in U \times I \times I} p(i >_u j |\Theta)^{\delta((u,i,j)\in D_S)} \\ \dot (1-p(i >_u j |\Theta))^{\delta((u,i,j)\notin D_S)}$$

위에서 함수 $\delta$는 아래와 같은 indicator 함수입니다.


$$\delta(b) := \left\{ {1\   \text{if b is true}\\ 0\   \text{else}} \right. $$
