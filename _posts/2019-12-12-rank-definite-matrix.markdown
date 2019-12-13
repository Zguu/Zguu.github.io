---
title: " [선형대수학] positive definite, semi-positive definite"
tags: LinearAlgebra Rank FullRank PositiveDefinite SemiPositiveDefinite
---

# Positive Definite 행렬
## 고유값으로 정의되지만..
***<center>A matrix is positive definite if it's symmetric and all its eigenvalues are positive</center>***
> 아주 간단한 정의이다. 행렬이 대칭행렬이고 고유값들이 모두 양수이면 된다고 한다. 하지만 여기서 바로 한가지 걱정이 생겨야 한다.
  아 고유값 저거 귀찮게 언제 다 계산하지?!

모든 eigenvalue를 계산하는 것은 matrix dimension이 증가함에 따라 복잡해진다. 당장 2x2 행렬에서 고유값 계산과 3x3 행렬에서 고유값 계산도 복잡도가 고꽤나 차이난다. dimension은 1 씩만 늘었는데.. 따라서 좀 더 효율적이고 덜 귀찮은 방법을 찾아야 한다.<br>
다음의 성질을 사용하자.
***<center>모든 eigenvalue의 부호는 pivot들의 부호와 같다.</center>***
> 3x3 행렬에서, pivot이 2,-3,3 으로 2개가 양수, 1개가 음수라면, 고유값 또한 2개는 양수이고 1개는 음수라는 성질이다. 해당 성질에 대한 증명은 (어렵다.)

위의 성질을 이용하면 처음 제시된 positive definite 정의를 다음과 같이 바꿀 수 있다.
***<center>A matrix is positive definite if it's symmetric and all its pivots are positive</center>***
> 해당 매트릭스가 symmetric이며, 모든 pivots value가 양수이면 positive definite matrix로 본다.

즉, pivot들의 부호만 확인하면 된다.

아래의 경우를 보면서 pivot을 통한 positive definite 확인을 해보자.
<center>$$\begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$$</center>
위의 행렬을 Gaussian elimination을 통해 변경하면 다음과 같다.
<center>$$\begin{pmatrix} 1 & 2 \\ 0 & -3 \end{pmatrix}$$</center>
행렬의 대각선에 있는 값들은 각각 1, -3 이며 해당 pivot 값들 중 1개는 양수이며 1개는 음수이다. 따라서 eigenvalue 또한 (우리가 계산은 아직 안해봤지만) 1개는 양수이고 1개는 음수임을 알 수 있다.

## pivot 계산도 쉽지가 않은데..?
$\ $k번째 pivot 값은 다음과 같이 쉽게 계산할 수 있다.
<center>$$d_k = \frac{det(A_k)}{det(A_{k-1})}$$</center>
여기에서 $$A_k$$는 upper left k x k submatrix에 해당한다. 다음 범위 $$1 \le k \le n$$ 에 해당하는 모든 $k$에 대하여 다음이 $$det(A_k)$$ 성립한다면 모든 pivot 값들은 양수임이 확인 될 것이다. 따라서 모든 submatrix 의 determinants 값이 양수임을 확인하면 된다. 아래의 행렬이 positive definite일지 계산해보자.
<center>$$\begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}$$</center>
<center>$$d_1 = 2$$</center>
<center>$$d_2 = \begin{vmatrix} 2 & -1 \\ -1 & 2 \end{vmatrix} = 3$$</center>
<center>$$d_3 = \begin{vmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{vmatrix} = 4$$</center>
$$d_1, d_2, d_3 > 0 $$ 이므로, positive definite 행렬임이 확인된다.

> 2x2 행렬 $$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$$ 에서 det() = ad-bc 로 쉽게 계산할 수 있다.<br>
 마찬가지로, 3x3 행렬 $$\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}$$ 에서 det() = a(ei-fh) - b(di-fg) + c(dh-eg)로 상대적으로 쉽게 계산 할 수 있다. 하지만, 행렬 차원수가 커짐에 따라 이러한 방법도 점점 복잡해진다.

## energy-based definition
