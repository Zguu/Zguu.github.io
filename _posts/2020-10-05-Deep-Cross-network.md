---
title: " [추천시스템] 논문리뷰 - Deep & Cross Network for Ad Click Predictions"
tags: RecommenderSystem
---

# ABSTRACT
Feature engineering은 중요하시만, 수작업으로 하기에는 한계가 있고 DNN이 그에 대안이 될 수 있다. 그러나, DNN은 모든 상호작용을 implicitly dustksgkau, 효율적이다고 볼수만은 없다. (모든 상호작용 텀을 모두 계산하기 때문)
DCN은 DNN의 일반적인 장점들을 유지하면서 더 나아가, 특정 범위 degree 내에서 특성 상호작용 효과를 명시적으로 학습한다.

# DEPP & CROSS NETWORK (DCN)
DCN의 아키텍쳐에 대해 알아보자. DCN 모델은 embedding and stacking layer로 시작하며, 이후에 cross network, deep network가 평행하게 추가된다. 마지막으로 combination layer에서 두 분리된 네트워크를 결합하는 방식이다. 해당 개념은 아래 Figure 1에 잘 나타난다.

![스크린샷 2020-10-05 오후 4.28.05](https://i.imgur.com/crXBnng.png)

## Embedding and Stacking layer
우리는 입력 데이터들이 sparse feature와 dense feature 둘 모두를 지닌 것으로 간주한다. 웹 규모 환경에서 이뤄지는 CTR 예측과 같은 추천 시스템에서, 입력 데이터들은 거의 대부분 범주형 feature 에 속한다. 이러한 feature 들은 보통 one-hot 벡터로 인코딩 된다. 그러나, 이것은 과도하게 고차원의 feature 공간을 사용해야만 하는 단점을 낳기 마련이다. <br>

이러한 고차원 벡터의 차원 수를 줄이기 위해, 이러한 binary feature들을 임베딩 하며, 이 결과로 실수 값들이 dense하게 구성하는 벡터를 얻을 수 있다. 이 벡터를 임베딩 벡터로 얘기한다.

$$\mathbf{x}_{embed, i} = \mathbf{W}_{embed,i}\mathbf{x}_i$$

여기에서, $\mathbf{x}_{embed,i}$는 임베딩 벡터이며, $\mathbf{x}_i$는 $i$번째 카범주에 있는 binary 입력 데이터이다. 그리고, $\mathbf{W}_{embed,i}\in \mathbb{R}^{n_e \times n_v}$ 는 임베딩 행렬에 속하며, 이 행렬은 전체 네트워크 내에서 다른 파라미터들과 함께 최적화 되야할 대상이 될 것이다. $n_e, n_v$는 각각 임베딩 사이즈와, vocabulary의 사이즈를 의미한다.<br>
이후에, 각각의 embedding 벡터들을 stacking 하여 아래와 같은 정제된 입력 데이터를 얻는다. $x_{dense}^T$는 원래 데이터를 normalize 한 dense feature 이다.

$$\mathbf{x}_0 = [ \mathbf{x}_{embed,1}^T,...,\mathbf{x}_{embed,k}^T,\mathbf{x}_{dense}^T ]$$

위의 $\mathbf{x}_0$ 벡터가 network로 feed 될 것이다.

## Cross Network
novel cross network의 핵심 아이디어는 명시적 feature crossing을 효과적인 방법으로 적용하는 것이다. cross network는 cross layer들로 이뤄져 있으며, 각각의 층은 다음과 같은 형식을 따른다.

$$ \mathbf{x}_{l+1} = \mathbf{x}_0\mathbf{x}_l^T\mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l = f(\mathbf{x}_l, \mathbf{w}_l, \mathbf{b}_l) + \mathbf{x}_l$$

여기에서, $\mathbf{x}_l, \mathbf{x}_{l+1} \in \mathbb{R}^d$ 들은 $l$번째와 $l+1$ 번째에 있는 cross layer의 출력값을 의미하는 열 벡터이다. 이와 비슷하게, $\mathbf{w}_l, \mathbf{b}_l \in \mathbb{R}^d$ 들은 각각 $l$번째에 있는 weight, bias paramter 들이다. 각각의 cross 층들은 입력으로 들어오는 벡터들을 연산하는 $f$ 함수 연산을 통해 다음 형태로 넘어간다. 이에 대한 시각적 표현은 아래 Figure 2와 같다.

![스크린샷 2020-10-05 오후 4.28.16](https://i.imgur.com/qAGwMx2.png)

# CROSS NETWORK ANALYSIS
