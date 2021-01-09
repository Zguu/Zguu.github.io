---
title: " [선형대수학] Principal Components and the Best Low Rank Matrix"
tags: LinearAlgebra
---

# PCA?

PCA는 가장 일반적으로 쓰이는 차원축소법 중 하나입니다. 계산 방식이 어렵지 않으며, 가장 기초적인 차원축소법이기 때문에 빠르게 데이터의 차원을 축소시키기 위해 많은 분야에서 사용됩니다.데이터의 dimension이 너무 높아서 분석에 연산량이 너무 많이 들 때에, 차원 수를 감소시키면 연산속도를 증가시킬 수 있습니다.
> 대부분의 모델에서 Column수가 수십만 or 수천만으로 매우 큰 경우, dimension 축소가 필수일 것입니다.

PCA의 princial component를 구해내는 방법에는,
- Lagrange multiplier를 이용해서 미분을 활용하는 방법
- SVD의 singular value를 활용하는 방법

크게 두가지가 있는데요, 선형대수학적 이해도를 높이기위해, SVD를 통한 PCA에 대한 접근을 진행해보겠습니다.

## SVD reminds

SVD를 통해 우리는 $A$ 행렬을 다음과 같이 표현했습니다.

$$ A = U\Sigma V $$

$A$ 행렬의 rank 가 $k$라고 할때, 위와 같이 행렬을 분해하면, $U, V$의 column 수는 $k$개, $\Sigma$의 0이 아닌 diagonal element 의 수 또한 $k$개가 됩니다. 따라서 아래와 같이 표현됨을 이미 공부한 바 있습니다.

$$ A_k = \sigma_1u_1v_1^T + \dots + \sigma_ku_kv_k^T$$

즉, 원래 행렬 $A$를 위와 같이 새로운 행렬 (또는 비슷한 행렬) $A_k$로 appoximation 하는 것입니다. $A_k$행렬은 rank 값이 $k$인 행렬입니다.

> matrix approximation, 즉 원래 함수와 최대한 비슷한 함수 $A_k$를 만들어내는 것입니다. matrix approximation 과정이 무엇인지 아직은 직관적으로 이해가 안될 수도 있습니다. 아래 matirx approximation에서 좀 더 깊이 다뤄 볼 것입니다.

PCA를 알아보려고 하는데, 왜 뜬금없이 SVD 이야기로 시작하는 것일까요? PCA의 주요 목적은 사실 matrix approximation 문제이기 때문입니다. matrix approximation에 대해 아래에서 조금 더 구체적으로 알아봅시다.

## Matrix approximation
rank 값이 $k$이며, 행렬 $A$와 가장 비슷한 행렬(closest rank k matrix to $A$)을 어떻게 구할 수 있을까요? 이에 대한 문제를 풀기 전에 우선 행렬들간의 유사도를 어떻게 측정할 것인지를 정의해야 합니다. 이에 대한 matrix norm 측정법에는 크게 세가지가 있습니다.

**Spectral norm** $||A||_2 = \text{max}\frac{||Ax||}{||x||} = \sigma_1$<br>
**Frobenius norm** $||A||_F = \sqrt{\sigma_1^2+\dots+\sigma_r^2}$<br>
**Nuclear norm** $||A||_N = \sigma_1 + \sigma_2 + \dots + \sigma_r$<br>

위의 세가지 norm으로 행렬의 크기를 구할 수 있습니다. 두 행렬간의 차이는 위의 셋중 하나 norm을 적용하여 아래와 같이 계산하면 될 것입니다.

$$||A-B||$$

예를 들어, $A = \begin{vmatrix} 4 & 0 \\ 0 & 2\end{vmatrix}$ 와 $B = \begin{vmatrix} 5 & 1 \\ 0 & 3\end{vmatrix}$ 두 행렬간의 차이에 대한 Frobenius norm 은,

$$ A-B = \begin{vmatrix} -1 & -1 \\ 0 & -1 \end{vmatrix} $$ 에서 $\sigma_1^2 + \sigma_2^2 = (-1)^2 + (-1)^2 = \sqrt{2}$ 가 됩니다.

이와 같이 norm을 계산하여, 각 행렬들 간의 차이를 수치화할 수 있으며, 이 차이가 가장 적은 행렬을 찾아내는 것이 matrix approximation의 목적이 됩니다.

> 추가로, 위의 세가지 norm 모두 unitarily invariant 가 성립합니다. $||Q_1A\bar{Q}_2^T|| = ||A||$ 즉, orthogonal 행렬을 좌우로 곱해도 norm 값은 변하지 않습니다. 따라서, $||A||$는 $\Sigma$로 부터 계산될 수 있습니다. unitarily invariant 특성을 우선 잘 기억해놓겠습니다. 아래에서 사용될 일이 있습니다.

## Eckart-Young Theorem : Best Approximiation by $A_k$

Eckart-Young Theorem 은 다음과 같습니다.
**Eckart-Young** if $B$ has rank $k$ then $||A-B|| \ge ||A-A_k||$

위의 정리가 의미하는 바는 다음과 같습니다. rank $k$ 인 모든 행렬들 $B$ 들에 대하여, 행렬 $A$와의 norm ($||A-B||$)은 무조건 행렬 $A$와 행렬 $A_k$의 norm ($||A-A_k||$)보다 크거나 같다는 것입니다.
 즉, A라는 행렬에 대하여, rank $k$인 행렬들 중에, norm 차이가 가장 적은 (가장 가까운 행렬)은 $A_k$라는 뜻입니다. 결과적으로 $A$ 행렬에 대한 rank $k$ 수준에서 best approximation matrix는 $A_k$라는 결론을 얻었습니다.
 예를 들어보겠습니다.

 $$\text{the rank two matrix closest to }A =  \begin{vmatrix} 4 & 0 & 0 & 0 \\ 0 & 3 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 1 \end{vmatrix} \text{is}  \quad  A_2 = \begin{vmatrix} 4 & 0 & 0 & 0 \\ 0 & 3 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{vmatrix}$$

Theorem에 따르면, 이것은 무조건 True 입니다. 이 diagnoal matrix가 너무 간단하다고 말할 수 있지만 unitarily invariant 의 특성과 연결지어 이해해보면, 어떠한 4 by 4 행렬이라도 singular values가 4,3,2,1인 경우라면 위의 행렬 $A$와 같은 상황으로 이해할 수 있습니다. <br>

$||A-A_2||$ 의 $L^2$ error 는  $2= max(2,1)$이고, Frobenius norm 은 $||A-A_2||_F = \sqrt{5}$ 입니다.

## 돌아와보면
PCA는 사실 best rank k approximation이다.
