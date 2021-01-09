---
title: " [추천시스템] 논문리뷰 - BPR: Bayesian Personalized Ranking from Implicit Feedback "
tags: RecommenderSystem
---

## 추천시스템에서 Ranking
추천 시스템에서, 특히 implicit feedback 데이터로 이뤄진 경우, 개인의 관심사에 대한 ranking을 산정하는 일반적인 접근법이 많이 존재하지 않는다. 이는, implicit feedback 데이터의 경우 explicit feedback 데이터와는 다르게, 대부분의 feedback 데이터가 0과 1로 이뤄져있기 때문에, 우리가 정확히 유저의 선호도를 예측하더라도 (해당 유저 $u$ 가 상품 $i$를 클릭할지($1$) 안할지($0$)에 대한 값을 모두 맞춘다고 하더라도), 우리가 정확하게 예측한 1 값들에 대하여 어떤 값이 더 정확하게 예측된 것인지, 우열을 가려내기가 힘들기 때문이다. 이에 대한 자세한 설명은 뒤에서 더 다뤄보도록 한다.

실제 추천시스템을 사용해야하는 많은 상황에서, 우리가 추천할 상품이나 서비스를 어떤 순서로 노출할 것 인지는 매우 중요하다. 검색 시스템에서도 검색 결과를 어떠한 순서로 노출해주느냐에 따라 유저의 반응도 또는 흥미의 정도가 다른 것과 마찬가지로, 추천시스템에서 노출 순서에 따른 유저의 반응 차이는 분명할 것이다. 따라서, 추천시스템이 추천하는 상품들의 결과를 잘 sorting 하는 것이 매우 중요한데, 이러한 방향에 집중한 랭킹 학습 모델(personalized ranking model)에 일반적인 접근법이 필요하다.

## introduction
이 논문은 개인화 랭킹에 대한 학습 모델에 일반적인 접근법을 제공한다. 저자가 제시하는 contributions은 다음과 같다.
1. 이 논문은 MPE 로부터 유도된 BPR-OPT 최적화 기준을 제시한다. 이 BPR-OPT 최적화는 ROC 커브 면적(AUC) 최대화 방법론과 유사함을 보일 것이다.
2. BPR-OPT 최대화를 위해, Stochastic gradient descent 방법론에 기반한 LearnBPR 이라는 학습 알고리즘을 제시한다. Stochastic gradient descent 알고리즘이 일반적인 full gradient descent에 비해 우월함을 보일 것이다.
3. 이 LearnBPR 학습 모델을 당시 SOTA 추천 모델 객체인 MF, adaptiveKNN에 적용해본다.
4. 다른 학습 모델에 디해 BPR 학습모델이 개인화 랭킹 학습에 있어서 더 좋은 성능을 보인다는 결과를 제공한다.

## Personalized Ranking
개인화 랭킹의 대표적인 예시는, 온라인 쇼핑몰에서 우리가 마주치는 상품들의 목록이다. 해당 상품들은 list로 정렬되어 우리에게 노출되는데, 이는 유저가 갖는 선호도가 높은 상품을 더 높은 곳에 노출시킨 결과일 것이다. 또한, 우리는 상품들에 대한 explicit feedback이 아닌, implicit feedback 상황을 가정하고 연구를 진행한다. 특정 상품에 대한 유저의 과거 구매유무 또는 클릭유무와 같은 데이터들은 간접적으로 유저가 해당 상품에 대한 선호도를 갖고있다고 추정할 수 있다. 하지만 이러한 implicit 데이터는 유저가 영화에 남기는 별점과 같은 explicit 데이터에 비해 선호도를 명확하게 반영한다고 보기는 논리적으로 어렵다. 또한, explicit 데이터의 경우, 유저가 해당 상품이나 서비스에 대한 부정적 rating을 남길 수도 있는데에 반해, implicit 데이터의 경우는 무조건 긍정적인 feedback만을 수집할 수 있다. 유저가 탐색한 상품을 모두 positive feedback으로 보는 데에도 어느 정도 논리적 허점이 있는데, 탐색하지 않은 상품이라고 해서 negative feedback이라고 결론 짓는 것은 상당한 논리적 비약에 해당할 것이다. 게다가 대부분의 commerce의 경우, 현실적으로 유저가 모두 탐색할 수 없을정도로 상품의 수는 무한하게 많은 데에 반해, 유저가 탐색하는 상품의 수는 극히 적다.
> 유저가 실제로 클릭해보고 구매하는 상품의 수는 제한적이다. 클릭하지 않은 상품들에 대하여 ''유저가 그 상품을 싫어해서 클릭하지 않았다' 라고 말하는 것은 논리비약이다.
### Formalization
$U$를 전체 유저 집합, $I$를 전체 상품 집합이라고 놓자. 아래에서 볼 수 있는 implicit dataset 은 $S \subseteq U \times I$ 에 해당한다. 유저 $u$가 특정 상품들 $i, j$에 대하여, 무엇을 더 선호하는지에 대한 정보를 다음과 같이 표현한다. $$ i\ >_u\ j$$

![스크린샷 2020-09-15 오후 1.14.05](https://i.imgur.com/gwHyA43.png)
위에서 상품 $i,\ j$ pair에 대해 유저 $u$가 무엇을 더 선호하는 지 비교선호도를 구할 수 있었는데, 같은 방법으로 전체 상품 pair에 대한 비교선호도를 구할 수 있다. 이러한 전체 pair에 대한 유저 $u$의 비교선호도를 간략하게 $>_u$로 표현하며 이렇게 비교선호도를 기록하는 $>_u$는 $I^2$에 포함된다. ($>_u \subset I^2$)
> 상품이 총 1000개라고 한다면, 우리는 1000개 상품과 1000개 상품들간의 비교 선호도를 기록해야 한다.

또한, 완벽하게 개인들에 대한 total order 를 제공해야 하며, total order의 성질은 다음과 같다.
1. $\forall i,j \in I : i \ne j \Rightarrow i >_u j \vee j >_u i\  (totality)$
2. $\forall i,j \in I : i >_u j \wedge j >_u i \Rightarrow i = j\  (antisymmetry)$
3. $\forall i,j,k \in I : i  >_u j \wedge j >_u k \Rightarrow i >_u k\ (transitivity)$

위의 Figure 1 에서 유저가 해당 상품에 대한 feedback 이 있는 경우는 $+$, 그렇지 않고 어떠한 feedback 도 없는 경우는 $?$ 로 표현이 돼있는데, 편의를 위해 feedback이 있는 경우의 데이터들을 다음과 같이 정의하자.
$$I_u^+ := \left\{ i \in I : (u, i) \in S\right\}$$
$$U_i^+ := \left\{ u \in U : (u, i) \in S\right\}$$
