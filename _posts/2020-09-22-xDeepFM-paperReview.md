---
title: " [추천시스템] 논문리뷰 - xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"
tags: RecommenderSystem
---

## Abstract
Combinatorial features (상호작용 특성)들은 많은 추천 시스템에서 중요한 역할을 해오고 있다. 상호작용 특성들은 모델에서 좋은 역할을 함에도 불구하고, web-scale system 에서 raw data 의 변동성, 크기, 속도 때문에 계산에 있어서 높은 비용을 수반하는 편이다. FM이나 DeepFM과 같은 Factorization 기반 모델들은 벡터의 내적 term으로 상호작용을 측정하며 이는 자동으로 상호작용 특성들의 패턴을 학습할 수 게하고, 보이지 않는 특성들에 대한 일반화 또한 가능하게 한다.
 이 논문은 Compressed Interaction Network(CIN) 을 제안하며, 이는 explicit 한 방법을 사용한 vector-wise 수준에서 feature interaction 생성을 가능하게 한다. 이는 Convolutional neural networks (CNNs), Recurrent neural networks(RNNs) 의 기능들을 모두 공유한다는 것을 보여줄 것이다.<br>
 더 나아가, 이 CIN 구조와 DNN을 결합하여 eXtreme Deep Factorization Machine 인 (xDeepFM)을 제시한다. 이 xDeepFM은 명시된 feature 상호작용에 대하여는 explicitly하게 학습이 가능하며, 반면에, arbitrary한 low, high order 특성 상호작용에 대하여는 implicitly한 학습을 보인다.

## Introduction
Data Science 에서 피쳐들의 상호작용 텀을 활용하는 것은 매우 일반적이지만, 이에 대한 disadvantage 들도 있다. 크게 세가지는 다음과 같다.
1. 좋은 퀄리티의 피쳐 상호작용을 잡아내는 데에는 그만큼 비용이 많이 든다.
2. 좋은 퀄리티의 피쳐 term을 잘 정제해낸다고 하더라도, 매우 많은 상호작용 term들이 생기는 경우가 일반적이며, 예측 모델에 과부하를 줄 수 있다.
3. manually or hand crafted 상호작용 term은 보이지 않는 상호작용 term을 찾아내는 데에는 한계가 있다. 따라서 DNN 등의 모델을 함께 사용해야 한다.

FM과 같은 모델은 $i$ 번째 피쳐들을 각각 latent factor vector $ v_i = [v_{i1}, v_{i2}, ... v_{id}]$ 로 임베딩을 진행하며, 이를 활용한 각 피쳐들의 상호작용 계산은 latent factor vector의 내적으로 진행된다. latent factor vector 내부의 $v_{i1}$과 같은 값들은 $bit$ term으로 칭한다. FM의 핵심인 상호작용 개념을 활용한 추가적인 논문들은 Factorization-machine supported Neural Network(FNN), Product-based Neural Network(FNN) 등이 있다. 이 모델들은 pre-trained FM을 활용하지 않는다는 것이 특징인데, high-order 피쳐 상호작용에 집중하고 있으며, low-order interaction에는 관심을 크게 주지 않는다. Wide & Deep, DeepFM 모델들은 이러한 앞 모델들의 단점을 hybrid 구조를 제시함으로써 극복했다.
이 xDeepFM 논문은 Deep & Cross Network (DCN)에 기반을 두고 접근하며, 이 모델은 기본적인 FM의 implicit interatcion module과, Wide & Deep, DeepFM 모델의 explicit high order interaction module 의 장점을 모두 결합한다. xDeepFM 의 contribution을 정리해보면 다음과 같다.

1. manual fature engineering 이 필요하지 않다.
2. bit-wise level의 피쳐 상호작용보다는 vector-wise level의 피쳐 상호작용을 활용한다.
3. 기존의 SOTA 알고리즘들보다 더 개선된 성능을 보여준다.

## Preliminaries
### 2.1 Embedding Layer
web scale 추천 시스템에서, 입력 피쳐들은 사이즈가 큰 차원에, sparse하며 spatial 또는 temporal correlation이 없는 특징을 지닌다. 그에 따라, $multi \ field$ 카테고리 형태가 일반적으로 사용된다. 이에 대한 예시 형태는 아래와 같다.

$$\underbrace{[0,1,0,0,...,0]}_\text{userid}\underbrace{[1,0]}_\text{gender}\underbrace{[0,1,0,0,...,0]}_\text{organization}\underbrace{[0,1,0,1,...,0]}_\text{interests}$$

위와 같은 one-hot encoding 형태의 데이터들은 임베딩 이후 사용된다. 임베딩 벡터 $\mathbf{e}$ 는 다음과 같이 표현된다.

$$\mathbf{e} = [\mathbf{e_1}, \mathbf{e_2}, ..., \mathbf{e_m}]$$
![스크린샷 2020-09-22 오후 2.42.54](https://i.imgur.com/LalQa11.png)
### 2.2 Implicit High-order Interactions
FNN, DeepCrossin 그리고 Wide & Deep 모델의 deep part 는 high-order 상호작용을 학습하기 위해 field embedding vector $\mathbf{e}$에 feed-forward nerual network를 활용한다. 우리가 잘 알고 있는 forward process는 다음과 같다.
$$ \mathbf{x}^1 = \sigma(\mathbf{W}^{(1)}\mathbf{e} + \mathbf{b}^1) $$
$$ \mathbf{x}^k = \sigma(\mathbf{W}^{(k)}\mathbf{k-1} + \mathbf{b}^k) $$

우리의 모델 구조는 아래 Figure 2에서 보이는 것에서 $FM \ or \ Product \ Layer$가 포함되지 않는 점만 제외하면 거의 비슷하다. 이 모델은 bit-wise 방식의 상호작용 term을 학습한다.
> bit-wise 방식의 상호작용을 학습한다는 것은, 같은 field의 embedding 결과인 벡터 내부 element 들이 서로 상호작용 영향을 갖는다는 것을 의미한다.


![스크린샷 2020-09-22 오후 2.43.06](https://i.imgur.com/IJImHaU.png)

### 2.3 Explicit High-order Interactions
아래의 Figure 3에서 CrossNet 아키텍쳐를 확인할 수 있다. 이 구조는 high order 피쳐 상호작용을 모델링한다. 전통적인 fully-connected feed-forward network와는 다르게, 해당 아키텍쳐의 hidden layer는 다음과 같이 계산된다.
$$ \mathbf{x_k} = \mathbf{x_0}\mathbf{x_{k-1}}^T \mathbf{w}_k + \mathbf{b_k} + \mathbf{x_{k-1}}$$
위와 같이 계산되는 각각의 $k$번째 hidden layer들은 첫번째 layer와 $k-1$ 번째 hidden layer의  곱을 통해 계산되므로, $\mathbf{x_0}$의 scalar 배수에 해당한다.
![스크린샷 2020-09-22 오후 2.43.17](https://i.imgur.com/3dlubXL.png)

CrossNet 아키텍쳐는 다른 DNN 모델들에 비할수 없을만큼 빠르고 효율적으로 피쳐 상호작용에 대한 효과를 학습하지만, 단점은 다음과 같다.
1. 각각의 hidden layer 가 $\mathbf{x_0}$ 의 배수인 만큼 복잡성이 제한된다.
2. 상호작용이 bit-wise 방식으로 진행된다.

## 3. Proposed model
### 3.1 Compressed Interaction network
이 논문이 제시하는 CIN 구조는 다음을 고려하여 디자인되었다.
1. 상호작용 효과가 bit-wise level이 아닌, vector-wise level에서 적용된다.
2. 높은 차원의 피쳐 상호작용들은 explicitly 하게 측정된다.
3. 네트워크의 복잡성이 상호작용의 degree와 함께 exponentially 증가하지 않아야한다.

임베딩 벡터들은 vector-wise 상호작용의 단위(unit)으로 여겨지므로, field 임베딩의 출력을 다음과 같이 표현하자.$$\mathbf{X}^0 \in \mathbb{R}^{m \times D}$$
그리고, $\mathbf{X}^0$의 $i$ 번째 행에 해당하는 값들은 $\mathbf{X}_{i,*}^0 = \mathbf{e}_i$ 로 표현한다. $D$는 각 임베딩 벡터의 차원을 의미한다 (임베딩 벡터들의 length). 또한, $k$번째 layer의 출력 값은 $$\mathbf{X}^k \in \mathbb{R}^{H_k \times D} $$로 표현할 수 있다. $H_k$는 $k$번째 layer의 필드 벡터 갯수를 의미한다. 각 층에 대하여 $\mathbf{X}_k$는 다음과 같이 계산될 수 있다.
$$\mathbf{X}_{h,*}^k = \sum_{i=1}^{H_{k-1}}\sum_{j=1}^{m} \mathbf{W}_{ij}^{k,h} (\mathbf{X}_{i,*}^{k-1} \circ \mathbf{X}_{j,*}^{0})$$
여기에서 $ 1 \le h \le H_k, \quad \mathbf{W}^{k,h} \in \mathbb{R}^{H_{k-1}\times m}$ 이며, $\mathbf{W}^{k,h}$는 $h$ 번째 feature vector 행렬에 대한 파라미터이다. $\circ$는 Hadamard product, 즉, element wise matrix product 이다. 위의 식에서 확인할 수 있듯이, $\mathbf{X}^k$는 $\mathbf{X}^{k-1}$과 $\mathbf{X}^0$ 사이의 상호작용을 통해 구해지며, 따라서, 피쳐 상호작용들은 explicitly 계산된다는 것을 알 수 있다. 또한, 상호작용의 degree 는 layer가 깊어질 수록 더욱 증가한다. 다음 은닉층이 가장 최근 은닉층과 추가되는 은닉층에 의존한다는 점에서, CIN 의 구조는 Recurrent Neural Network (RNN)의 구조와 매우 흡사하다.<br>
 위의 식은 CNN 과도 구조적으로 유사하다는 점은 흥미로운데, 우리가 $(\mathbf{X}^{k-1} \circ \mathbf{X}^{0})$ 에 해당하는 부분을 intermediate tensor $\mathbf{Z}^{k+1}$로 생각한다면, 이 $\mathbf{Z}^{k+1}$은 하나의 이미지로, $\mathbf{W}^{k,h}$는 하나의 필터로 생각될 수 있다.

아래 이미지에서 볼 수 있듯이, (a) 에서 각 행렬의 outer products를 통해 intermediate tensor를 생성하고, (b)에서 이 tensor 들에 필터를 적용해 다음 층의 행렬을 생성하는 것으로 이해해볼 수 있다.

 ![스크린샷 2020-09-22 오후 2.43.30](https://i.imgur.com/PmOHQqk.png)

위의 Figure 4(c)에서 CIN의 전체 구조를 확인할 수 있는데, 모든 은닉층 $\mathbf{X}^k, k\in [1,T]$ 는 출력 unit과 연결 돼있으며, 우리는 일차적으로 은닉층의 각 feature map 에 대한 sum pooling을 진행한다. $$p_i^k = \sum_{j=1}^D \mathbf{X}_{i,j}^k$$
$i$ 값은 1과 $H_k$ 사이에 존재할 것이다. 그러므로, 우리는 다음과 같은 pooling vector를 얻는다. $\mathbf{p}^k = [p_1^k, p_2^k, ..., p_{H_k}^k]$ rk <br>
각 층들에 대한 모든 $\mathbf{p}^k$는 다음과 같이 concatenated 된다. $\mathbf{p}^+ = [\mathbf{p}^1,\mathbf{p}^2, ... ,\mathbf{p}^T] \in \mathbb{R}^{\sum_{i=1}^T H_i}$. 우리가 CIN 구조를 바로 binary classification에 사용하게 된다면, output unit 은 $\mathbf{p}^+$에 sigmoid 함수를 적용한 아래의 형태가 된다.
$$ y = \frac{1}{1 + exp(\mathbf{p}^{+T})\mathbf{w}^0}$$

### 3.2 CIN Analysis
해당 섹션은 CIN의 space complexity, times complexity, polynomial appoximation 에 대한 분석을 진행하고 있다. 이에 대한 내용은 paper ```Deep & Cross Network for Ad Click Predictions (2017)``` 을 함께 읽으며 추후에 정리해보도록 한다.

### 3.3 Combination with Implicit Networks
2.2 섹션에서 논의한 것과 마찬가지로, plain DNN 구조들은 Implicit high-order 피쳐 상호작용들에 대한 학습을 진행한다. CIN 과 plain DNN을 서로 약점을 보완해주는 관계가 될 수 있으며, 이러한 각자의 장점들 때문에, 이 둘을 함께 조합하여 사용하면 더 좋은 결과를 낼 수 있지 않을까 하는 호기심을 가질 수 있다. 우리의 xDeepFM 모델 구조는 아래에 나와있으며 Wide & Deep or DeepFM 모델과 구조적으로 매우 유사하다. xDeepFM은 저차원과 고차원의 피쳐 상호작용을 모두 포함할 뿐만 아니라, implicit, explicit feature 상호작용을 모두 포함한다. 출력 값은 다음과 같이 간단하다.
$$ \hat{y} = \sigma(\mathbf{w}_{linear}^T\mathbf{a} + \mathbf{w}_{dnn}^T \mathbf{x}_{dnn}^{T} + \mathbf{w}_{cin}^T \mathbf{p}^+ + b)$$
binary classification의 경우에, loss function은 또한 우리에게 친숙한 다음과 같은 log loss일 것이다.
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N y_i log\hat{y}_i + (1-y_i)log(1-\hat{y}_i)$$
최종적으로 해당 loss $\mathcal{L}$에 regularization term한 추가하여 최적화를 진행한다.
$$\mathcal{J} = \mathcal{L} + \lambda_{*}||\mathsf{\Theta}||$$

![스크린샷 2020-09-22 오후 2.43.43](https://i.imgur.com/x4Ae1aj.png)
