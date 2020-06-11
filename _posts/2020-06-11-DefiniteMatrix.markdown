---
title: " [선형대수학] Review on Symmetric Positive Definite Matrix"
tags: LinearAlgebra PositiveDefinite SemiPositiveDefinite
---

\usepackage[shortlabels]{enumitem}

# Symmetric Positive Definite matrix
$$\href{https://zguu.github.io/2019/12/12/rank-definite-matrix.html}{이전 포스팅}$$의 포스팅에서  이미 positive matrix에 대한 다양한 정와 계산 예시를 정리한 적이 있다. 강의를 들으며 추가로 얻은 정보들이 있어서 이번 포스팅에서 한 번 더 정리하자.
$\ $Symmetric Positive definite 행렬을 정의할 수 있는 test 방법은 크게 다섯가지가 있다. 아래에서 $\mathbf{S}$는 Symmetric matrix를 가르킨다.

## 5 tests

(1) 행렬 \mathbf{S}의 모든 eigenvalues는 0보다 커야한다.
(2) Energy based 정의해 의해 \mathbf{x}^T\mathbf{Sx} > 0 이 성립해야 한다.
(3) \mathbf{S} = \mathbf{A}^T\mathbf{A} 가 성립하며, \mathbf{A} 에 모든 column은 서로 독립이다.
(4) 모든 leading Determinants 는 0보다 커야한다.
(5) All pivots in elimination 은 0보다 커야한다.

위의 총 5개에 해당하는 positive definite test는 이 중 하나라도 성립하면 나머지 test도 자동으로 통과하는 등치관계에 있다. <br>
$$\mathbf{S} = \begin{bmatrix} 3 & 4 \\ 4 & 6 \end{bmatrix}$$ 행렬을 보자. 이 행렬의 determinant det($\mathbf{S}$) 는  15 - 16 = -1 으로 음수이다. 행렬의 determinant 는 eigenvalue들의 곱과 같으며, 따라서 이 행렬의 eigenvalue들이 2개일 때, 이 두 값의 곱은 음수임을 의미한다. 즉 두 개 eigenvalue 가 양수일 수 없으므로, **test 1** 에 의해 이 행렬은 positive definite matrix로 볼 수 없다.<br>

**test4** 에서 leading determiants는 $\mathbf{S}$ 의 왼쪽 상단부터, 1 x 1, 2 x 2 size window matrix 의 determinant를 가르킨다. ($$\begin{bmatrix} 3 \end{bmatrix}, \begin{bmatrix} 3 & 4 \\ 4 & 6 \end{bmatrix}$$) <br>

**test5** 의 pivots 관점에서 보면, $\mathbf{S}$의 first pivot은 좌측 최상단 원소에 해당하는 3이다. 이 행렬 $\mathbf{S}$를 elimination form으로 변경하면 $$\begin{bmatrix} 3 & 4 \\ 0 & \frac{2}{3} \end{bmatrix}$$ 이며, 여기서 diagonal line의 두번째 원소인 $$\frac{2}{3}$$이 second pivot에 해당한다. 또한 이는 $$\frac{2x2 size Matrix Determinant}{1x1 size Matrix Determinant}$$ 에 해당한다. 즉, **test4** 와 **test5** 는 사실 같은 test 라는 것을 알 수 있다.
이는 $$\href{https://zguu.github.io/2019/12/12/rank-definite-matrix.html}{이전 포스팅}$$ 에 조금 더 예시와 함께 잘 설명 돼있다.

Gilbert 교수님은 **test2** 가 positive definite matrix 의 정의에 가장 잘 부합하는 (자신의 주관적 생각으로) 정의에 가까운 테스트 법이라고 소개했다.
