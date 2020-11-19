---
title: " [추천시스템] Factorization Machine"
tags: RecommenderSystem
---

# Factorization + Machine ..?
Factorizaion Machine (이하 FM) 논문은 2010년 Steffen Rendle 교수님의 IEEE conference 논문에서 발표됐습니다. 현재까지 추천시스템에서 큰 영향을 끼치고 있는 연구이며, FM의 핵심개념은 Deep Learning과 결합해, 추후에 DeepFM, xDeepFM 등의 연구로도 발전했습니다. <br>
Rendle 교수님의 또다른 역작으로는 Bayesian Personalized Learning(BPR) 논문이 있으며, FM과 BPR 모두 간단해 보이는 아이디어를 획기적인 방법으로 적용한다는 공통점이 있습니다. 또한, 두 아이디어 모두 행렬 분해의 개념에서 많은 영향을 받은 것으로 보입니다. FM 논문을 자칫 잘못 이해하게 되면, Linear Regression과 큰 차이가 없다고 느낄 수도 있습니다. 특히 interaction term이 추가된 polynomial 형태와 상당히 유사하다고 느낄 수도 있습니다. 하지만 이 연구의 이름에서 느껴지듯이 **Factorizaion** 에 연구의 핵심이 있습니다.

그에 따라 연구의 제목도 Factorization 으로 시작하게 되는데, 아무래도 이후에 붙는 Machine은 SGD 최적화 과정을 반영하여 붙인 이름인 듯 합니다.

> 핵심은 Factorization이다.

## Dataset
거두절미하고, 저자가 제시한 데이터셋과 이 모델의 목적에 대해서 먼저 알아보겠습니다.

![스크린샷 2020-11-19 오후 7.26.26](https://i.imgur.com/M6tM3vJ.png)

위의 데이터셋에는 Feature Vector에 해당하는 $\mathbf{x}$ 와 Target scalar $y$가 있습니다. 파란색, 주황색, 갈색의 프레임 안에 속하는 데이터들은 one-hot encoding으로 표현된 categorical 변수들이고, 노란색에 해당하는 정규화된 벡터 변수입니다. 초록색에 속하는 시간 변수는 categorical 변수임에도 불구하고 one-hot encoding을 하진 않았네요. 아무래도 이 시간이라는 변수는 연속형 변수로 볼 수도 있고, 범주형 변수로 볼 수도 있는 특성이라 그런 것 같습니다. 연구자가 어떻게 사용함에 따라 달라질 수 있는 부분이겠죠. <br>

## Goal
이러한 각각의 벡터 데이터를 이용해서 우리가 해야할 것은, 해당 유저가 (파란색의 1에 속하는 유저) 특정 영화 (주황색의 1에 속하는 영화) 에 대하여 평점을 몇점 남겼는지 예측하는 것입니다. 유저가 총 100명, 영화가 50개, 다른 영화도 50개, 시간은 1개의 변수, 마지막으로 본 영화를 30개 변수로 놓는다면, 사실 이 문제는 231개의 독립변수를 이용하여, 평점이라는 종속변수를 예측해내는 Regression 문제로 볼 수 있습니다. <br>
일반적인 선형회귀적 접근으로 이 문제를 바라보게 되면, 아쉬운 점이 많이 보일 수 밖에 없습니다. 특히 바로 개선이 필요하다고 느낄 수 부분은 interaction term에 관한 것일 겁니다.

## Interaction
![interaction](https://i.imgur.com/294Uddx.png)

위의 데이터를 보게되면 우리는 다음과 같은 의구심을 가질 수 있습니다.
> " A라는 유저는 B라는 유저와 평점을 매기는 패턴이 비슷한데, 그렇다면 A가 보지 않은 영화에 대해 B가 본 경우, B의 평점을 참조하면 되지 않을까? "

> " Titanic 영화를 오후 20시에 본 사람들이 평점을 보통 높게 주는 경향이 있네..."

즉, 2개 이상의 변수들이 동시에 작용하는 경우 (항상 둘 다 1인 경우가 아닌)를 고려한 Regression은 더욱 좋은 결과를 낼 수 있다는 것입니다. 그런데 사실 이 개념은 iteraction term을 추가한 회귀 문제에 불과하지 않는데요, FM 연구가 2010년 이후로 꾸준히 주목을 받고있는 것은 이 interaction term을 Factorization으로 접근해서 풀어냈다는 것입니다. 이 부분이 잘 이해가 안될수도 있고 아리송하실 텐데 실제 수식과 행렬을 보며 이해해봅시다.

## Equations

FM 모델의 수식은 다음과 같습니다.

$$\hat{y(\mathbf{x})} := w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n <\mathbf{v}_i, \mathbf{v}_j> x_i x_j \quad (1)$$

$w_0$값은 절편에 해당하는 scalar, $\mathbf{w}$ 는 weights vector에 해당하며, $\mathbf{V}$는 각 변수들 간의 interaction을 계산할 수 있도록 factorized 된 형태의 행렬입니다. 이 행렬 $\mathbf{V}$를 이용해 변수들 간의 interaction term을 구해냅니다. 즉, 상호작용을 구해내기 위해서, 1차적으로 factorized된 형태로 상호작용을 준비하는 재료기록용 행렬이라 생각하면 될 것 같습니다.
현재 우리는 총 $n$개의 독립변수가 있다고 가정할 것이며, weight에 해당하는 $\mathbf{w}$의 길이도 그에 따라 $n$이 될 것입니다. 또한, 모든 변수들간의 상호작용을 기록하는 행렬 $\mathbf{V}$는 각 변수들을 총 $k$ 개의 factor로 표현할 예정이므로, $n \times k$ 형태가 됩니다. 정리해보면 다음과 같습니다.

$$ w_0 \in \mathbb{R},\  \mathbf{w} \in \mathbb{R}^n, \  \mathbf{V} \in \mathbb{R}^{n \times k} \quad (2) $$

$<\mathbf{v}_i, \mathbf{v}_j>$ 는 두 벡터 간의 내적을 의미합니다.

예시를 들어보면 다음과 같습니다.

$$\mathbf{X} = \begin{bmatrix} 1 & 0 & 0 & 1 & 1 & 0 & 0.3 \\
0 & 1 & 0 & 1 & 1 & 0 & 0.1 \\&&&\vdots \\ 0 & 0 & 0 & 1 & 1 & 0 & 0.2 \end{bmatrix}$$


$$ \mathbf{x_1} = \begin{bmatrix} 1 & 0 & 0 & 1 & 1 & 0 & 0.3 \end{bmatrix}$$
$$ \mathbf{x_2} = \begin{bmatrix} 0 & 1 & 0 & 1 & 1 & 0 & 0.1 \end{bmatrix}$$

위와 같이, input에 해당하는 변수들이 행렬과 각각의 벡터로 표현될 수 있습니다. 현재의 예시에서 우리는 총 7개의 변수를 갖고 있습니다. 그렇다면 weight에 해당하는 $\mathbf{w}$ 벡터의 길이는 7이 될 것입니다.

$$ \mathbf{w} = \begin{bmatrix} w_1, w_2, \cdots w_7 \end{bmatrix}$$

그리고, 변수들 간의 상호작용 term을 기록할 $\mathbf{V}$ 행렬은 총 7개의 행, $k$개의 열이 될 것입니다. 여기에서, $k$는 factor의 수가 되며, 임의로 3이라고 두도록 하겠습니다.

$$\mathbf{V} = \begin{bmatrix} 0.31 & 0.11 & -0.1 \\ -0.23 & -0.9 & 0.27\\ &\vdots \\ 1.0 & -0.91 & 0.34 \end{bmatrix}$$

위의 7 $\times$ 3 행렬을 이용해서 이해 각 변수들간의 모든 상호작용 term을 한 행렬안에 계산 해 넣을 수 있습니다. 이 행렬의 이름은 $\mathbf{W}$로 놓겠습니다. $\mathbf{V}\mathbf{V}^T$로 계산합니다.

$$\mathbf{W} = \mathbf{V}\mathbf{V}^T$$

모든 7개의 변수들간의 상호작용 term을 7 $\times$ 7 행렬에 넣어두게 됐습니다. 이후에 이 상호작용 행렬에 있는 값들을 참고하여, 상호작용 항을 계산할 수 있습니다. 앞선 식에서 보았던 항 $<\mathbf{v}_i, \mathbf{v}_j> \mathbf{x}_i \mathbf{x}_j$는 $V_{ij} X_{i,} X_{j,}$에 해당합니다.

이제 우리가 해야할 것은, 각 parameter 벡터와 행렬에 해당하는 $\mathbf{w}, \mathbf{W}$의 모든 미지수들을 최적화하는 것입니다. 최적화의 방향은 목표로 하는 값 (여기서는 평점)이 실제 평점과는 최대한 차이가 적도록 하는 것입니다.

## Complexity

위의 모든 파라미터들을 최적화하는 과정에 있어서, 파라미터의 수는 변수 $\mathbf{x}$의 길이에 제곱해서 늘어날 수 밖에 없습니다. 즉, $O(kn^2)$의 연산 복잡도를 지니게 됩니다. 하지만, 위에서 $\mathbf{V}$행렬 내의 모든 내적 연산을 간단히 해보면 $O(kn)$으로 복잡도를 획기적으로 줄일 수 있습니다. 이에 대한 증명은 저자가 잘 제공하고 있습니다. 그리 어렵지 않으므로 차근차근 따라가보는 것을 추천합니다.

$$\sum_{i=1}^n \sum_{j=i+1}^n <\mathbf{v}_i, \mathbf{v}_j> x_i x_j \\ = \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n <\mathbf{v}_i, \mathbf{v}_j> x_i x_j - \frac{1}{2} \sum_{i=1}^n <\mathbf{v}_i, \mathbf{v}_i> x_i x_i \\
= \frac{1}{2}(\sum_{i=1}^n\sum_{j=1}^n\sum_{f=1}^k v_{i,f}v_{j,k}x_i x_j - \sum_{i=1}^n\sum_{f=1}^k v_{i,f} v_{i,f} x_i x_i)\\
=\frac{1}{2}\sum_{f=1}^k (( \sum_{i=1}^n v_{i,f} x_i) (\sum_{j=1}^n v_{i,f}x_j) - \sum_{i=1}^n v_{i,f}^2 x_i ^2) \\
= \frac{1}{2}\sum_{f=1}^k ((\sum_{i=1}^n v_{i,f} x_i) ^2 -\sum_{i=1}^n v_{i,f}^2 x_i^2$$

또한, 추천시스템의 대부분의 벡터변수들은 우리가 위의 dataset figure에서 보았던 것처럼 대부분 0인 데이터가 많으며 (high sparsity), 이러한 특성 때문에 계산 복잡도는 실제로 더 줄어들게 됩니다.
사실 여기까지만 이해하면, tensorflow model subclassing 을 통해, custom function으로 문제를 쉽게 풀어낼 수 있습니다.

## Applications

우리가 처음 제시했던 dataset에서 1~5에 해당하는 영화평점을 예측하는 regression 문제였지만, 우리가 target data와 학습과정에서의 loss function 을 어떻게 설정하느냐에 따라 이 모델은 충분히 classifier로도 작용할 수 있습니다.

> 또한, 우리가 training dataset 에서 target $y$들을 유사도 순서 또는 클릭 순서등의 순서 데이터를 넣게되면 ranking 을 학습할 수 있습니다.

## Learning

${\partial\over\partial \theta} \hat{y(\mathbf{x})} = \left\{ {1, \quad if\  \ \theta\ \  is\ \  w_0 \\ x_i, \quad if\ \  \theta\ \  is\ \   w_i \\ x_i \sum_{j=1}^n v_{j,f} x_j - v_{i,f} x_i^2, \quad if\  \theta\ \  is\ \  v_{i,f}}  \right.$

위의 미분 식을 따라가며,stochastice graident descent (SGD) 방식으로 학습합니다. tensorflow 2.0 에서 ```tf.keras.optimizer.SGD``` 를 사용합니다.

## n-way FM
지금까지 우리가 봤던 FM 모델은 두개 변수들간의 모든 상호작용을 고려한 2-way FM 모델입니다. 조금 더 일반화 된 d-way FM 모델의 형태는 아래와 같습니다.

$$ \hat{y(x)} := w_o + \sum_{i=1}^n w_i x_i \\
 \sum_{l=2}^d \sum_{i_1=1}^n cdots \sum_{i_l = i_{l-1}+1}^n (\prod_{j=1}^l x_{i_j}) (\sum_{f=1}^{k_l}  \prod_{j=1}^l v_{ {i_j}, f}^l)$$

$l$번째 상호작용 텀 파라미터에 해당하는 변수들은 PARAFAC 모델에 의해 factorized 됩니다. 각 행렬과 변수는 아래와 같습니다.

$$\mathbf{V}^{(l)} \in \mathbb{R} ^ {n\times k_l},\quad k_l \in \mathbb{N}_0^+$$

## 결론 & 느낀점
이후 저자가 제시하는 이 모델에 대한 정당화 파트는 다루지 않겠습니다. SVM 과의 비교를 주로 하고있습니다. 사실 우리가 이 모델의 구조를 제대로 이해하면 SVM이 해결하지 못하는 부분에 적용할 수 있겠다 라는 생각이 직관적으로 들게 됩니다. 마지막으로 이 논문의 장점과 시사하는 바, 기억해야할 부분을 정리하며 포스팅을 마치겠습니다.

- FM 모델은 선형 회귀 모델에 interaction 텀을 추가하여, 보이지 않는 factor 효과를 반영합니다.
- interaction 텀 계산 과정에 있어서, factorization을 이용한 연산법은 기본적인 interaction term을 뛰어넘는 효과를 발휘합니다
- 이 모델의 구조와 구현은 tensorflow 2.0 으로 간단히 해결할 수 있으며, 모델의 복잡도는 변수의 수 $n$, factor의 수 $k$의 곱에 선형 비례합니다.

사실 이 논문의 아이디어는 매우 간단하며, 대부분의 Data scientist 가 추천 뿐 아니라 회귀 문제에서 맞닥뜨릴 수 있는 문제들에 대해 쉽게 해결책이으로 떠올릴 수 있는 것이라 생각합니다.  저도 실제로 회귀 문제를 푸는 데에 있어서 interaciton term들의 weight를 최적화할 아이디어를 찾다가 FM 모델에 다시 당도하게 됐는데요. <br> 하지만 저자의 천재성이 돋보이는 부분은 Factorization Part를 추가하여 추가적인 성능 개선을 이끌어냈다는 점. 그리고 실제로 추천 시스템에서 이것이 10년 넘게 영향력을 행사오고 있다는 점. 마지막으로 연산 복잡도까지 간단하게 풀어낸 점이라고 생각합니다. 시간이 되면 코드 구현 리뷰도 추가하도록 하겠습니다.
> 나도 잘하고싶다~

## Reference
Fatorization Machines (S Rendle , IEEE International Conference on Data Mining, 2010)
