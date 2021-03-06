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


## Positive Definite Matrices and Minimum Problems

$$\begin{bmatrix}\mathbf{x} & \mathbf{y}\end{bmatrix}\begin{bmatrix}3 & 4 \\ 4 & 6\end{bmatrix}\begin{bmatrix}\mathbf{x} \\ \mathbf{y} \end{bmatrix} = f(\mathbf{x}, \mathbf{y})$$

$$\begin{bmatrix}\mathbf{x} & \mathbf{y}\end{bmatrix}\begin{bmatrix} 3\mathbf{x} + 4\mathbf{y} \\ 4\mathbf{x} + 6\mathbf{y}\end{bmatrix} \\ = 3\mathbf{x}^2 + 4\mathbf{xy} + \mathbf{xy} + 6\mathbf{y}^2 $$
$$ = 3\mathbf{x}^2 + 8\mathbf{xy} + 6\mathbf{y}^2 $$ (quadratic energy form)<br>
이에 해당하는 그래프는 아래와 같다.
<center><img src="https://imgur.com/jOCYLWv.png" width="60%" height="60%"></center>

대부분의 Deep Learning, Machine Learning, Neural Nets, Big Computation에서 손실함수 (loss function)들은 대부분 위와 같은 Bowl Shape 형태를 기본으로 보인다. 즉, 이렇게 생긴 energy 함수를 minimize 시키는 데에 목적이 있다고 볼 수 있다.
$$f(\mathbf{x},\mathbf{y}) = \mathbf{xS}\mathbf{x}^T + \mathbf{x}^T\mathbf{b}$$ 그래프는 아래와 같다.
<center><img src="https://imgur.com/9gs19Sb.png" width="60%" height="60%"></center>
위의 두 Bowl Shape 모두 Convex 형태에 해당하며, 많은 실전 문제들은 어떻게 이 Convex의 특정지점에서 최소 지점으로 갈 것인지를 다룬다. 즉, Gradient 를 계산하는 문제이다. 최솟값을 찾을 수 있는 함수의 형태는 strictly convex 이어야만 한다.<br>
**그렇다면, 해당 함수가 convex 인지 아닌지 어떻게 판단할 것인가?**
일반적으로, 하나의 변수 $x$ 에 대한 함수 $$f(x)$$ 에 대한 최솟값 존재 유무는 아래와 같이 확인할 수 있다. <br><br>

<center> $x = x_0$ 에서, 도함수 ${df \over dx} = 0 $을 만족하고, <br> 이계도함수 ${d^2f \over dx^2}$를 만족하는 지점은 최소점에 해당한다.</center><br>

하지만 단일 변수가 아니라, 다변수 함수인 경우 문제는 조금 더 복잡해지며, 이계 도함수들을 행렬로 표현해야 한다. 이때, 이 이계도함수 행렬이 positive definite를 만족하는 경우, 우리는 해당 지점을 최솟값으로 볼 수 있다. 정리하자면 아래와 같다.<br><br>

<center> 점 $x_0, y_0$ 에서, 도함수 $\partial{f}\over\partial{x}$$ $$ =  0$, $\partial{f}\over\partial{y}$ $= 0$ 을 만족하고, <br>

이계도함수 행렬 $\begin{bmatrix} \partial^2 f \over \partial x^2 & \partial^2 f \over \partial x \partial y \\ \partial^2 f \over \partial x \partial y & \partial^2 f \over \partial y^2\end{bmatrix}$ 이 positive definite 인 경우, <br>
해당 점 $x_0, y_0$ 에서 이 함수는 최솟값을 갖는다고 말할 수 있다. </center><br>

Gradient Descent 에서, 각각의 step은 steepest한 방향으로 발을 뻗어나가며 가장 낮은 지점 $$x^{*}$$ 를 찾아낸다. 이를 caculus, linear algebra의 문법으로 표현하면 각각 아래와 같다.<br>

- **Calculus** The partia derivatives of $$f$$ are all zero at $$x^*$$ : $$\partial f \over \partial x_i $$ $ = 0$
- **Linear Algebra** The matrix $$S$$ of second derivatives $$\partial^2 f \over \partial x_i \partial x_j $$ is positive definite


우리는 여기서 Gradient Descent 에 대해 깊게 다루고자 하는 것은 아니다. Eigenvalue 값의 차이가 Gradient Descent 알고리즘에서 어떤 영향을 끼치는 지에 대한 힌트만 얻고 넘어가기로 한다.<br>

> 뭐...간단히 정리해보면 positive definite matrix는 해당 행렬이 수렴 가능한 convex 인지 아닌지에 대한 힌트를 주므로 중요하다.
