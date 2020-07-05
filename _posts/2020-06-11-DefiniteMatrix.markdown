---
title: " [선형대수학] Review on Symmetric Positive Definite Matrix"
tags: LinearAlgebra
---

# Symmetric Positive Definite matrix
$$\href{https://zguu.github.io/2019/12/12/rank-definite-matrix.html}{이전 포스팅}$$의 포스팅에서  이미 positive matrix에 대한 다양한 정와 계산 예시를 정리한 적이 있다. 강의를 들으며 추가로 얻은 정보들이 있어서 이번 포스팅에서 한 번 더 정리하자.
Symmetric Positive definite 행렬을 정의할 수 있는 test 방법은 크게 다섯가지가 있다. 아래에서 $\mathbf{S}$는 Symmetric matrix를 가르킨다.

## 5 tests

>(1) 행렬 $$\mathbf{S}$$의 모든 eigenvalues는 0보다 커야한다.
>
>(2) Energy based 정의해 의해 $$\mathbf{x}^T\mathbf{Sx} > 0$$ 이 성립해야 한다.
>
>(3) $$\mathbf{S} = \mathbf{A}^T\mathbf{A}$$ 가 성립하며, $\mathbf{A}$ 에 모든 column은 서로 독립이다.
>
>(4) 모든 leading Determinants 는 0보다 커야한다.
>
>(5) All pivots in elimination 은 0보다 커야한다.

위의 총 5개에 해당하는 positive definite test는 이 중 하나라도 성립하면 나머지 test도 자동으로 통과하는 등치관계에 있다. <br>
$$\mathbf{S} = \begin{bmatrix} 3 & 4 \\ 4 & 6 \end{bmatrix}$$ 행렬을 보자. 이 행렬의 determinant det($\mathbf{S}$) 는  15 - 16 = -1 으로 음수이다. 행렬의 determinant 는 eigenvalue들의 곱과 같으며, 따라서 이 행렬의 eigenvalue들이 2개일 때, 이 두 값의 곱은 음수임을 의미한다. 즉 두 개 eigenvalue 가 양수일 수 없으므로, **test 1** 에 의해 이 행렬은 positive definite matrix로 볼 수 없다.<br>

**test4** 에서 leading determiants는 $\mathbf{S}$ 의 왼쪽 상단부터, 1 x 1, 2 x 2 size window matrix 의 determinant를 가르킨다. <br>
($$\begin{bmatrix} 3 \end{bmatrix}, \begin{bmatrix} 3 & 4 \\ 4 & 6 \end{bmatrix}$$) <br>

**test5** 의 pivots 관점에서 보면, $\mathbf{S}$의 first pivot은 좌측 최상단 원소에 해당하는 3이다. 이 행렬 $\mathbf{S}$를 elimination form으로 변경하면 $$\begin{bmatrix} 3 & 4 \\ 0 & \frac{2}{3} \end{bmatrix}$$ 이며, 여기서 diagonal line의 두번째 원소인 $$\frac{2}{3}$$이 second pivot에 해당한다. 또한 이는 $$\frac{2\times2\   Matrix's\  Determinant}{1\times1\   Matrix's\ Determinant}$$ 에 해당한다. 즉, **test4** 와 **test5** 는 사실 같은 test 라는 것을 알 수 있다.
이는 $$\href{https://zguu.github.io/2019/12/12/rank-definite-matrix.html}{이전 포스팅}$$ 에 조금 더 예시와 함께 잘 설명 돼있다.

Gilbert 교수님은 **test2** 가 positive definite matrix 의 정의에 가장 잘 부합하는 (자신의 주관적 생각으로) 정의에 가까운 테스트 법이라고 소개했다. 아래와 같은 상황을 보자 <br>


## 잠시 Gradient Descent로 빠지는 교수님

**이 부분은 가볍게 읽어 넘긴다.**

$$\begin{bmatrix}\mathbf{x} & \mathbf{y}\end{bmatrix}\begin{bmatrix}3 & 4 \\ 4 & 6\end{bmatrix}\begin{bmatrix}\mathbf{x} \\ \mathbf{y} \end{bmatrix} = f(\mathbf{x}, \mathbf{y})$$
$$\begin{bmatrix}\mathbf{x} & \mathbf{y}\end{bmatrix}\begin{bmatrix} 3\mathbf{x} + 4\mathbf{y} \\ 4\mathbf{x} + 6\mathbf{y}\end{bmatrix} = 3\mathbf{x}^2 + 4\mathbf{xy} + \mathbf{xy} + 6\mathbf{y}^2$$
$$ = 3\mathbf{x}^2 + 8\mathbf{xy} + 6\mathbf{y}^2 $$ (quadratic energy form)<br>
이에 해당하는 그래프는 아래와 같다.
<center><img src="https://imgur.com/jOCYLWv.png" width="60%" height="60%"></center>

대부분의 Deep Learning, Machine Learning, Neural Nets, Big Computation에서 손실함수 (loss function)들은 대부분 위와 같은 Bowl Shape 형태를 기본으로 보인다. 즉, 이렇게 생긴 energy 함수를 minimize 시키는 데에 목적이 있다고 볼 수 있다.
$$f(\mathbf{x},\mathbf{y}) = \mathbf{xS}\mathbf{x}^T + \mathbf{x}^T\mathbf{b}$$ 그래프는 아래와 같다.
<center><img src="https://imgur.com/9gs19Sb.png" width="60%" height="60%"></center>
위의 두 Bowl Shape 모두 Convex 형태에 해당하며, 많은 실전 문제들은 어떻게 이 Convex의 특정지점에서 최소 지점으로 갈 것인지를 다룬다. 즉, Gradient 를 계산하는 문제이다. 우리는 여기서 Gradient Descent 에 대해 깊게 다루고자 하는 것은 아니다. Eigenvalue 값의 차이가 Gradient Descent 알고리즘에서 어떤 영향을 끼치는 지에 대한 힌트만 얻고 넘어가기로 한다.<br>
<center>$$\begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y}\end{bmatrix} = \nabla f$$</center>

Gradient Descent 에서 가장 유의해야할 문제 중 하나는, 해당 Convex가 너무 narrow할 때, step size에 민감하게 수렴도가 바뀐다는 것이다. 즉 너무 뾰족하게 좁은 Bowl 모양에서 너무 큰 step size를 가져가면 우리는 최솟점에 도달하기 힘들 수 있다. <br>
즉, 이런 Bowl Shape 함수의 수렴 가능 여부는 Convex 의 narrow 정도에 영향을 받는데, eigenvalue는 해당 convex shape에 대한 단서를 준다.
> 뭐...간단히 정리해보면 positive definite matrix는 수렴 가능한 convex 형태를 주므로 중요하다.

## Definite matrix 의 유용한 성질들
Positive definite matrix에 해당하는 각 행렬 $$\mathbf{S, T}$$ 를 더해도 여전히 positive definite matrix에 해당할까? Energy test를 활용해서 증명해보자.<br>
$$\mathbf{S + T}$$가 positive definite 임을 증명하려면, 아래가 성립해야 한다. <br>
<center> $$\mathbf{x(S+T)x^T} > 0\ , \  for\   every\   \mathbf{x}$$
