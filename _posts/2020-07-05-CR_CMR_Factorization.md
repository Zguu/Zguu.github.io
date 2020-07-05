---
title: " [선형대수학] CR, CMR 분해의 중요성"
tags: LinearAlgebra
---

# CR, CMR Factorization
CR, CRM 분해가 중요한 것은, 이것이 사이즈가 큰 행렬들에게 중요하기 때문이다. $C$ 행렬의 열들은 A의 열들로부터 직접적으로 구해지고 (데이터 값이 변하지 않고), 마찬가지로 R의 행들이 A로부터 직접적으로 구해진다. 즉, 데이터의 유실이 일어나지 않는다. 더욱 유명하고 잘 알려진 분해 방법인 QR 분해, SVD는 분해는 그 과정에서 원래 행렬의 데이터 값들을 손실하거나 특성을 잃을 수도 있는데, CR 분해와 CMR 분해는 그렇지 않다는 점에서 간단하지만 유용하다. $$A = QR,  A = U\Sigma V^T$$ 는 분해 과정에서 vector들 orthogonalizing 하는 과정을 거치지만, $$C,\ R$$은 원래 데이터를 잃지않고 유지할 수 있다.

> 만약 $A$가 양수행렬이라면, $$C, R$$도 마찬가지이다.

> 만약 $A$가 sparse 한 행렬에 해당한다면, $$C, R$$도 마찬가지이다.

아래에서 예시를 통해 더 직관적으로 살펴보자
## CR Factorization
A 행렬은 C, R로 분해가 가능하다. A = CR
C는 A행렬의 basis of column space로 구성 돼있으며, 따라서 C 행렬의 컬럼은 모두 독립이다. 마찬가지로 R은 A행렬의 basis of row space로 구성 돼 있으며, 또한 R 행렬의 행은 모두 독립이다. 해당 행렬의 랭크는 컬럼 스페이스의 차원 수와 같다.
> The rank of a matrix is the dimension of its column space

예를 들어, $$ A = \begin{bmatrix} 2 & 4 \\ 3 & 6 \end{bmatrix} $$ 일 때, 이 행렬은 다음과 같이 분해된다.
$$ A = \begin{bmatrix} 2 & 4 \\ 3 & 6 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}\begin{bmatrix} 1 & 2 \end{bmatrix} $$
행렬 $$A$$는 rank 가 1인 행렬 (두번째 행렬은 첫번째 행렬에 dependent 이므로) 이기 때문에, 행렬 $$C$$는 diemension이 1이며, $$R$$또한 마찬가지이다. 

R = rref(A) = row-reduced echelon form of A (without zero rows)
rank theorem : Column Rank = Row Rank (the number of independent columns equals the number of independent rows)


## CMR Factorization
A행렬을 C,R 로 분해하는 것에서 더 나아가, A = CMR 형태로 분해가 가능하다.
이는 mixing matrix M을 활용해서 진행한다. M 값이 스칼라 형태를 취할 수도 있지만, 대부분은 그렇지 않다.
M 은 다음과 같이 구할 수 있다.
$$A = CMR$$,
$$then,\   C^TAR^T = C^TCMRR^T$$
$$then,\   M = (C^TC)^{-1}(C^TAR^T)(RR^T)^{-1}$$

예시 추가.
