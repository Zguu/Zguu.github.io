---
title: " [선형대수학] Review on Symmetric Positive Definite Matrix"
tags: LinearAlgebra PositiveDefinite SemiPositiveDefinite
---


# Symmetric Positive Definite matrix
$$\href{https://zguu.github.io/2019/12/12/rank-definite-matrix.html}{이전 포스팅}$$의 포스팅에서  이미 positive matrix에 대한 다양한 정와 계산 예시를 정리한 적이 있다. 강의를 들으며 추가로 얻은 정보들이 있어서 이번 포스팅에서 한 번 더 정리하자.
$\ $Symmetric Positive definite 행렬을 정의할 수 있는 test 방법은 크게 다섯가지가 있다. 아래에서 $\mathbf{S}$는 Symmetric matrix를 가르킨다.

## 5 tests
$$\begin{enumerate}
  \item 행렬 \mathbf{S}의 모든 eigenvalues는 0보다 커야한다.
  \item Energy based 정의해 의해 \mathbf{x}^T\mathbf{Sx} > 0 이 성립해야 한다.
  \item \mathbf{S} = \mathbf{A}^T\mathbf{A} 가 성립하며, \mathbf{A} 에 모든 column은 서로 독립이다.
  \item 모든 leading Determinants 는 0보다 커야한다.
  \item All pivots in elimination 은 0보다 커야한다.
\end{enumerate}
$$

위의 총 5개에 해당하는 positive definite test는 이 중 하나라도 성립하면 나머지 test도 자동으로 통과하는 등치관계에 있다. <br>
$$\mathbf{S} = \begin{bmatrix} 3 & 4 \\ 4 & 6 \end{bmatrix}$$ 행렬을 보자. 이 행렬의 determinant det($\mathbf{S}$) 는  15 - 16 = -1 으로 음수이다. 행렬의 determinant 는 eigenvalue들의 곱과 같으며, 따라서 이 행렬의 eigenvalue들이 2개일 때, 이 두 값의 곱은 음수임을 의미한다. 즉 두 개 eigenvalue 가 양수일 수 없으므로, **test 1** 에 의해 이 행렬은 positive definite matrix로 볼 수 없다.
