---
title: " [추천시스템] Improving complementary-product recommendations"
tags: recommendations
---

# Complementary-product recommendations
amazon Science 블로그에 2020년 10월 7일에 업로드 된 글입니다. 보충재 상품을 추천하는 알고리즘 개발에 있어서 평균 정확도를 7% 증가시킨 접근법이라고 소개 돼있습니다.

다양한 추천 알고리즘들 중에 보충재 상품을 발굴하고 추천하는 방법론들에 대한 연구나 문서들은 상대적으로 그 양이 많지 않습니다. 보충재 상품 추천에 참고할 만한 글을 찾다가, 직접 찾은 글은 아니지만, 읽어볼만한 글이라고 소개받아서 간략하게 포스팅으로 정리합니다.

## Intro
complementary-product recommendation (CPR) 에 7% 개선된 성능을 보여준 알고리즘을 소개하도록 하겠습니다. 이는 [new deep-learning-based method](https://www.amazon.science/publications/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation) 로 [Conference on Information and Knowledge Management](https://www.amazon.science/conferences-and-events/cikm-2020)에서 발표 됐다고 합니다.

저자가 말하는 새로운 방법론은 크게 세가지 전략을 통해 개선 됐습니다.
1. **Better selection of training data for the CPR model**
2. **Greater diversity in the types of products recommended**
3. **Respect for the asymmetry of the CPR problem**

> SD카드는 카메라에게 좋은 보충재가 될 수 있지만, 카메라는 SD카드에게 좋은 보충재가 아닐 수 있습니다. 위와 같은 이슈를 비대칭 (asymmetry)라고 칭합니다.

또한, 이 방법론은 cold start의 문제를 다룰 수 있으며, 모델이 학습된 이후에 추가된 새로운 상품에 대한 보충재를 예측해 내는 데에도 사용될 수 있습니다. Amazon에서 개발된 embedding 방법론 Product2Vec을 활용하여, CPR 모델에 대한 input을 나타냈습니다.

## Implicit signals

다른 CPR 모델들과 아르지 않에 이번 모델에서도 training data는 implicit signal을 활용합니다. $x$ 상품이 다른 상품 $y$와 관련이 있다는 것은 다음과 같이 세가지 방법으로 포착해볼 수 있습니다.

1. co-purchase : $x$ 상품을 산 유저가 $y$라는 상품도 샀다.
2. co-view : $x$ 상품을 조회한 유저가 $y$라는 상품도 조회했다.
3. purchase after view : $x$ 상품을 조회한 유저가 결국에 $y$라는 상품을 구매했다.

![table1](https://i.imgur.com/6tyrXP4.png)

아래에서부터는 편의를 위해, 위의 각 세가지 데이터를 각각 CP, CV, PV로 표기하겠습니다.

일반적으로 CPR 모델들은 CV, PV를 유사도의 지표로 보며, CP를 complementarity의 지표로 사용합니다. 그러나 이 세가지 지표들은 서로 간에 상당히 겹치는 부분들이 존재합니다.

저자가 사용한 직관은 다음과 같습니다.
$$"\text{CP 데이터에는 존재했던 상품 pair이지만 CV와 PV에는 없는 상품 pair 라면,
 이는 더욱 좋은 예측에 사용될 수 있지 않을까?}"$$

CP에는 존재하지만 CV와 PV에는 존재하지 않는 상품 pair를 **co-purchase only(CP only)** 데이터로 부르겠습니다.

> 사실 이 부분이 뭐가 그렇게 대단한 직관인지 나로써는 잘 모르겠다. CP에서 존재하는 상품 pair, CV 상품 pair, PV 상품 pair 이 세가지 모두 보충재를 어느정도 예측해내는 데에는 다 일조를 할 것이다. 물론 CP가 가장 강력한 데이터로 작용할 것이라고도 생각이 된다. <br>

> 다만, CP는 예측해내지만 CV가 훌륭하게 예측해내는 결과들 & PV가 잡아내는 결과들. 이 모든 결과들을 잘 조합해서 CP, CV, PV를 모두 넘어서는 게 정말 대단한 것 아닌가? 아이러니하네


모델에 사용된 input은 Product2Vec embeding vector들이다. Product2Vec이 다른 embedding scheme 들과 다른 점은, 이것의 input들이 grpah 라는 점입니다. 즉 데이터 구조가 node (node가 상품 정보를 나타낸다)와 edge(CP, CV와 같은 상품들 간의 관계를 나타낸다)들로 이뤄져있습니다.

## Behavior-based Product Graph

아래의 Figure 2에 **Behavior-based Product Graph (BPG)** 에 대한 전체적인 그래프 구조를 확인 할 수 있습니다.

![스크린샷 2021-02-25 오전 8.49.01](https://i.imgur.com/1alDYrX.png)

$\mathcal{I}$ 는 product/item 의 집합, $\mathcal{C_i}$는 item $i$의 catalog feature (상품 카테고리, 타입, 제목, 설명과 같은) 라고 정의합니다.

$\mathcal{B} \in \mathcal{I} \times \mathcal{I}$는 상품들 간의 관계를 나타내는 테이블이 됩니다. 이 테이블은 총 세가지 종류가 존재하게 됩니다($\mathcal{B_{\text{CP}}}, \mathcal{B_{CV}}, \mathcal{B_{PV}}$)

예를 들어, 어떤 item $i \in \mathcal{I}$ 는 상품 타입 $w_i \in \mathcal{C}_i$ 에 속하며, 이는 hdmi-dvi-cable 또는 over-ear-headphone 과 같은 것입니다.

이러한 정보들을 **BPG** 형태로 표현 했을 때, 각각의 상품들은 "nodes", 상품의 타입이나 다른 카탈로그 feature 들은 "node attributes", 상품 간의 관계를 나타내는 데이터는 "edges" 에 해당합니다. 그로므로, BPG는 multi-relational attributed information network로 이해할 수 있습니다.

**Problem Formulation**

보충재 추천의 문제 정의는 다음과 같이 할 수 있습니다. 우리에게 상품 feuatres $\mathcal{C}$ 와 유저 행동 데이터 $\mathcal{B}$를 사용해서, 모델 $\mathcal{M}$을 학습해야 합니다. 이 모델은 상품 $i$가 상품의 type $w_i$와 diversity degree $K$가 함께 주어졌을 때 $K$개의 distinct 한 보충재 상품 type ${w_k, k \in {1, ..., K}}$ 을 예측해야 합니다.

이후에는 이렇게 예측된 보충재 상품 type을 사용해서 $K$개의 item 집합 ${S_{w_k}}$를 생성해야 하며, 해당 상품 pair의 동시구매 확률인 $\sum_{k=1}^K \sum_{j \in S_{{w}_k}} \mathbb{P}_{cp}(i,j)$ 를 최적화하는 방향으로 진행합니다.

## Diversification

CPR 모델은 input 상품에 대하여 가장 빈번하게 함꼐 구매된 상품을 출력하도록 학습됩니다. 그러나 이 방식은 출력 상품들의 homogeneity 한 결과 (다양성이 떨어지는)를 낳게 됩니다. 예를 들어, 테니스 라켓과 함께 가장 잘 팔린 상품은 브랜드가 다른 세 개의 테니스 공일 수 있습니다. 하지만, 실제로 유저가 원하는 보충재의 결과는 더욱 다양한 보충재 상품 군일 것입니다. 예를 들어, 테니스 라켓에 대한 top 3 추천 목록은 테니스 공, 오버그립 팩, 헤드핸드 와 같은 것과 같이 다양해야 합니다.

## Modeling
