---
title: " [논문리뷰] Knowledge-aware Complementary Product Representation Learning"
tags: Paper Review
---

# 보충재보충재 ComplementaryComplementary
각 상품에 대한 보충재와 대체재 추천은 쉬우면서도 어려운 문제다. 간단하게 추천 상품 목록을 뽑아낼 수 있는데, 이렇게 간단한 로직으로 생성해낸 보충재 및 대체재 목록이 꽤나 잘 먹힐 것이라는 생각이 든다.
Basket analysis 에서 기본으로 사용하는 support, confidence, lift values 만 잘 활용하거나, a priori 알고리즘을 사용하는 선에서 보통 수긍할 만한 보충재들을 얻을 수 있다. item2vec 논문의 개념에서 소개된 sequence 기반 임베딩 학습으로 상품들 간의 대체재 또한 쉽게 구해낼 수 있다. 첫 해결이 쉬우면서도 어느정도 성능을 보이게 된다면, 적은 일로 높은 효율을 내는 상황이기에, 기쁜 마음으로 칼퇴를 할 수 있다. 하지만 문제는 그 다음인데, 이 간단한 접근 방식의 추천 시스템의 퍼포먼스를 능가할 다른 알고리즘 개발은 어떻게 접근을 시작할 것이냐는 것이다. 상품 정보만 활용하는 것이 아니라, 유저나 환경에 대한 정보들, 즉 contextual data를 추가로 활용하면 더 낫지 않을까 하는 궁금증에 이것저것 찾아보다가, 이 논문을 접하게 돼서 리뷰를 해본다.

> 어떻게 더 좋은 보충재 목록을 구할 수 있을까?
> 어떻게 더 유사한 대체재들의 목록을 구할 수 있을까?

# INTRODUCTION
 요즘의 소비자들은 온라인 쇼핑몰에서 특정 카테고리에서만 쇼핑을 하는 것이 아니라, 다양한 카테고리를 드나들며 쇼핑을 진행한다. 이와 같은 이유로, 상품들간의 내제적 관계를 이해하는 것 뿐 아니라, 개개인 유저들의 선호 또한 고려하는것이 중요하다. 추천 시스템 중에서도, 보충재 추천 영역에서는 특별히 짚고 넘어가야하는 특성들이 있다. 보충재 관계에 있어서 알려진 특성들은 다음과 같다.
> A$\rightarrow$ B는 A 상품이 B 상품에 대한 보충재임을 의미한다.

- Asymmetric (비대칭) : HDMI $\rightarrow$ TV 는 성립하지만, TV $\rightarrow$ HDMI는 성립하지 않는다.
- Non-transitive (전이불가) : HDMI $\rightarrow$ TV, cable adoptor $\rightarrow$ HDMI 가 성립할 지라도, cable adoptor $\rightarrow$ TV는 성립하지 않는다.
- Transductive (확장성) : HDMI $\rightarrow$ TV가 성립할 때에, 다른 비슷한 TV들에 대해서도 HDMI는 보충재 역할을 수행한다.
- Higher-order (조합) : (TV, HDMI, cable) 조합에 대한 보충재는 각 개별 상품들의 보충재들과 다를 수 있다.

> 논문 중간중간 나오는 representation learning은 feature learning 과 동일한 의미이다.

유저들의 집합은 $\mathcal{U}$ 로 표기하며, 상품들의 집합은 $\mathcal{I}$로 표기한다. $U \in \mathcal{U}$ 는 유저를 표기하는 카테고리 변수이다. $\left\{ I_{t-1}, ...,I_{t-k} \right\}$ 는 $k$개의 연속하게 구매된 상품들의 집합이다. 우리가 다음과 같은 조건부 확률을 softmax 분류기와 score function $S(.)$를 이용해 계산한다면,

$$p(I_{t+1}|U,I_t,...,I_{t-k})$$

다음의 score function : $S(I_{t+1},(U,I_t,...,I_{t-k}))$은 user-item preference term 및 item complementary pattern term 으로 구분되야만 한다.

$$ S(I_{t+1}, (U,I_t,...,I_{t-k})) = f_{UI} (I_{t+1}, U) + f_I(I_{t+1}, I_t, ..., I_{t-k}) $$

위의 식에서 $f_UI(.)$ 부분은 유저의 상품에 대한 선호 (bias)를 의미하며, $f_I(.)$ 부분은 상품의 보충재 패턴에 대한 강도를 의미한다. 구매 시퀀스의 보충재 패턴이 약하다면, 즉, $f_I(I_{t+1},I_t,...,I_{t-k})$ 이 값이 작다면, 모델은 유저의 상품 선호도 텀을 더욱 크게 반영할 것이고, 반대도 마찬가지다.

> 이전의 다른 많은 논문들에서 contextual information이 퍼포먼스 향상에 많은 도움을 주었다고 갑자기 강조한다. 뜬금없이.. (알겠어;;))

# METHOD

$\mathcal{U}$ 는 N명의 유저들이 있는 집합이며, $\mathcal{I}$는 M개의 상품들이 있는 집합이다. 상품의 브랜드, 제목, 설명과 같은 contextual knowledge feature들은 tokenized 된 단어와 같이 이해할 수 있다. 마찬가지로 상품의 카테고리나 discretized 된 연속 변수 feature 들 역시 token으로 이해될 수 있다. 각각의 상품들에 대한 이 두 feature를 concatenate 하여, 하나의 벡터 $\mathbf{w}_i$로 표현하자. $\mathcal{W}$ 는 전체 토큰들이 이루고 있는 집합이며, 총 $n_w$개의 수를 갖는다. 유저 $u$에 대하여, 시점 $t$를 기준으로 이전 $k$개의 연속적인 구매를 종합하여 깔끔한 데이터 셋을 완성하면 다음과 같이 표현된다.

$$\left\{ u, i_t, i_{t-1},..., i_{t-k}, \mathbf{x}_u, \mathbf{w}_{i_t}, \mathbf{w}_{i_{t-1}}, ..., \mathbf{w}_{i_{t-k}} \right\}$$

## Bayesian Factorization
유저에 대한 정보와 유저가 최근에 구입했던 상품들에 대한 정보를 종합하여 다음 구매를 예측하도록 모델은 최적화된다. contextual knowledge를 item/user representation으로 인코딩하기 위해 다음과 같이 contextual knowledge prediction 을 진행한다.
- $p(\mathbf{I}_t|\mathbf{U},\mathbf{I}_{t-1},...,\mathbf{I}_{t-k})$ : 유저와 가장 최근 $k$의 주문 상품들이 주어졌을 때, 다음 구매할 상품을 예측한다.
- $p(\mathbf{W}_I|\mathbf{I})$ : 상품의 contextual knowledge feature를 예측한다. 각 상품들의 contextual feature들은 서로 독립이라고 가정하기 때문에, 다음과 같이 각각의 곱으로 표현할 수 있다.
$$p(\mathbf{W}_I|I) = \prod_{\mathbf{W}\in\mathbf{W}_I} p(\mathbf{W}|I)$$
- $p(\mathbf{X}_U|U)$ : 유저들의 feature를 예측한다. 이 항 또한 다음과 같이 곱으로 표현한다.
$$\prod_{\mathbf{X}\in\mathbf{X}_U} p(\mathbf{X}|U)$$

위의 세 항을 모두 함께 표현하게 되면, 아래 Table 1의 오른쪽에 있는 proposed model 의 Bayesian network representation과 동일함을 알 수 있다.

$$ log\ p(\mathbf{U},\mathbf{I}_t, \mathbf{I}_{t-1}, ..., \mathbf{I}_{t-k}, \mathbf{X}_U, \mathbf{W}_I, \mathbf{W}_{I_{t-1}},...,\mathbf{W}_{I_{t-k}}) $$
$$ = log\ \left\{ p(\mathbf{I}_t|\mathbf{U}, \mathbf{I}_{t-1}, .., \mathbf{I}_{t-k})p(\mathbf{X}_U|\mathbf{U})\prod_{j=0}^k log\ p(\mathbf{W}_{I_{t-j}|\mathbf{I}_{t-j}})  \right\} + C$$

$$ = log\ p(\mathbf{I}_t | \mathbf{U}, \mathbf{I}_{t-1},...,\mathbf{I}_{t-k})$$
$$ + \sum_{\mathbf{X}\in\mathbf{X}_U}\ log\ p(\mathbf{X}|\mathbf{U}) + \underbrace{\sum_{j=0}^k\sum_{\mathbf{W}\in\mathbf{W}_{I_{t-j}}}log\ p(\mathbf{W}|\mathbf{I}_{t-j})}_\text{Contextual knowledge} + C $$



![스크린샷 2020-10-20 오후 4.29.43](https://i.imgur.com/faSFdeF.png)
