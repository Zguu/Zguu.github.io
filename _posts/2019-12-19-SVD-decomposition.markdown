---
title: " [선형대수학] SVD (Singular Value Decomposition)"
tags: LinearAlgebra
---

# SVD (Singular Value Decomposition)
$ \$LU 행렬분해는 하나의 행렬을 두개의 행렬로 분해했고, LDU 행렬분해는 하나의 행렬을 세개의 행렬로 분해했었다. SVD는 고유값 분해 (EigenValue Decomposition)의 한계점을 극복할 수 있기에 유용하다. 고유값 분해는 일반적으로 정사각 행렬에만 적용이 가능한 데에 반해, SVD는 직사각행렬의 분해에도 사용이 될 수 있으며, 다양한 형태의 추가적인 분해가 가능하다.
## Unitary Matrix, Conjugate Transpose
$\ $ SVD 연산에 있어서 각 행렬들이 갖는 특징 중 대표적인 것이 Unitary or orthogonal 행렬들이 존재한다는 것, 그리고 각 행렬들의 Conjugate transpose를 빈번히 계산해야한다는 것이다. 사실 이 개념들에 대해 몰라도 연산을 따라오는 데에 큰 문제는 없으나, 엄밀한 SVD 의 정의가 이 두 개념을 포함해야만 하므로, 간단하게 설명한다.
### Unitary Matrix
아주 간단하게 말해서, $$AA^T = I$$ 를 만족하는 A 를 unitary matrix 라고 한다. 자기 자신의 전치행렬 transpose가 역행렬인 경우 $$A^T = A^{-1}$$, 해당 행렬은 unitary matrix 라고 칭한다. Orthogonal Matrix 와 같은 형태이며, 해당 행렬에 대한 예시는 다음과 같다.
#### Examples
<center>$$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$</center>
<center>$$\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$</center>
<center>$$\begin{bmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$</center>
<center>$$\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$</center>
### Conjugate Transpose
$\ $Conjugate Transpose는 말 그대로 행렬을 전치(transpose)해주고, 해당 복소수(complex)들을 그에 대한 conjugate 값으로 대체해주는 것이다. ***Hermitian conjugate, bedaggered matrix, adjoint matrix, transjugate*** 등 다양한 이름으로 불린다고 한다. 표기 또한 다양하며 아래의 예시를 보면 쉽게 어떤 개념인지 확인할 수 있다. ***bedaggered matrix*** 라는 이름은 아무래도 이 행렬을 표기할 때에 dagger 모양을 이용해서 표현하기 때문에 그런듯 하다 ($$ U^{\dagger}$$). Conjugate Transpose 행렬에 대한 정의는 다음과 같이 표기한다.<br>
<center>$$\mathbf{A}^H = (\bar{\mathbf{A}})^T = \bar({\mathbf{A}^T})$$</center>
예시는 다음과 같다.
<center> $$ A = \begin{bmatrix} 1 & -2-i & 5 \\ 1 + i & i & 4 - 2i \end{bmatrix}$$</center>
행렬 $A$가 위와 같다면, 그에 대한 tranpose는 다음과 같다.
<center> $$ A^T = \begin{bmatrix} 1 & 1+i \\ -2-i & i \\ 5 & 4-2i \end{bmatrix}$$</center>
tranpose 이후에 각 원소들을 conjugate 값으로 바꾼다.
<center> $$ A^H = \begin{bmatrix} 1 & 1-i \\ -2+i & -i \\ 5 & 4+2i \end{bmatrix}$$</center>
## SVD
$\ $ SVD 행렬 분해는 $M$라는 행렬을 $$U\Sigma V^T$$ 총 3개의 행렬로 분해한다. $$m \times n$$ 형태의 행렬 $M$이 있을 때, 이 행렬을 $$m \times m$$ 모양인 행렬 $U$, $$m \times n$$ 모양인 행렬 $\Sigma$, 마지막으로 $$n \times n$$ 모양인 행렬 $$V^T$$ 총 3개로 분해하게 된다. 여기에서 $$\Sigma$$ 행렬을 가운데로 왼쪽과 오른쪽에 위치하는 $$M$$, $$V^T$$행렬을 각각 ***left-singular vectors***, ***right-singular vectors*** 로 지칭한다. <br>
$\ $뒤에서 다시 예시를 보며 다루겠지만, $$U$$, $$V^T$$ 행렬은 모두 unitary matrix에 해당한다.(아래 이미지 참고) <br>
<center><img src="https://imgur.com/LR6wFKb.png" width="50%" height="50%"></center>

또한, $$\Sigma$$ 행렬은 정사각형의 diagonal matrix 이며, diagonal 원소들은 모두 0 이상의 값을 갖는다. 실제 행렬을 SVD 방법론으로 분해해보면서, $$M, \Sigma, V^T$$ 행렬의 각 계산법에 대해 먼저 익히고, SVD 가 갖는 의미에 대해 살펴보자.
> real matrix 에서는 특정 행렬($$A$$)의 전치 행렬 (transpose)를 $$A^T$$로 표현하지만, real matrix를 확장한 complex matrix에서는 conjugate transpose의 개념을 적용해야하며, 이에 대한 표기는 다양하나 일반적으로 $$A^{* \ast}, A^{\dagger}$$와 같이 표기하는 것이 맞다. 하지만 이번 포스팅에서는 편의를 위해 일반적인 전치 행렬 표현 $$A^T$$를 주로 사용할 것이다.

1. 행렬 $M$이 주어졌을 때, $U$ 행렬은 $$MM^T$$의 orthonormal eigenvector 이다.
2. 행렬 $M$이 주어졌을 때, $V$ 행렬은 $$M^TM$$의 orthonormal eigenvector 이다.
3. $$MM^T$$와, $$M^TM$$에서 각각 얻은 고유값들 중에 양수인 값들의 ***루트값*** 을 이용해 diagonal한 matrix를 생성했을 때, 해당 행렬이 $$\Sigma$$ 이다.
> ex) $$MM^T$$의 고유값이 (2,8,0), $$M^TM$$의 고유값이 (2,8) 이라면, 두 고유값 집합에 모두 포함되는 2와 8을 이용해 diagonal matrix를 만든다. 만약 이 diagonal matrix의 사이즈가 2 초과라면, 다른 대각행렬의 원소들은 0으로 채운다. $$\begin{bmatrix} \sqrt{2} & 0 & 0 \\ 0 & \sqrt{8} & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

$\ $실제 행렬을 보며 위의 계산 방법을 적용해보자.
<center>$$ M = \begin{bmatrix} 0 & 1 & 1 \\ \sqrt{2} & 2 & 0 \\ 0 & 1 & 1 \end{bmatrix}$$</center>
해당 행렬의 전치행렬을 우측에 곱해주면 아래와 같은 값을 얻는다.
<center>$$ MM^T = \begin{bmatrix} 2 & 2 & 2 \\ 2 & 6 & 2 \\ 2 & 2 & 2 \end{bmatrix}$$</center>

위의 1번 $U$에 대한 정의를 참고해보면, 위에서 얻은 $$MM^T$$행렬의 eigenvector들을 구해야 한다. 고유값과 고유벡터에 대한 식 $$A\mathbf{x} = \lambda \mathbf{x}$$, $$ det(A-\lambda I) = 0 $$을 이용해서 고유값 $\lambda$와 고유벡터 $x$들을 구할 수 있다. 이 방법을 통해 얻은 $\lambda$에 대한 다항방정식은 아래와 같다.

<center>$$-\lambda^3 + 10\lambda^2 - 16\lambda = -\lambda(\lambda^2 - 10\lambda + 16)$$</center>
<center>$$ = -\lambda(\lambda - 8 )(\lambda - 2)$$</center>
<center>$$\therefore \lambda = 8, 2, 0$$</center>
<center>Also... singular values are $$\sigma_1 = 2\sqrt{2}, \sigma_2 = \sqrt{2}, \sigma_3 = 0$$</center>

$$\lambda = 8$$일 때, eigenvector $$\mathbf{x}_1$$은 $$(\frac{1}{\sqrt{6}},\frac{2}{\sqrt{6}} ,\frac{1}{\sqrt{6}})$$ 이다. 마찬가지로, $$\lambda = 2$$일 때, eigenvector $$\mathbf{x}_2$$은 $$(-\frac{1}{\sqrt{3}},\frac{1}{\sqrt{3}} ,1\frac{1}{\sqrt{3}})$$ , $$\lambda = 0$$일 때, eigenvector $$\mathbf{x}_3$$은 $$(\frac{1}{\sqrt{2}},0 ,-\frac{1}{\sqrt{2}})$$ 이다. 따라서, 이를 종합해 우리가 원하는 ***left-singular vectors*** 는 다음과 같은 행렬 형태로 표현한다.
<center>$$ U = \begin{bmatrix} \frac{1}{\sqrt{6}} &  -\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} \\ \frac{2}{\sqrt{6}} & \frac{1}{\sqrt{3}} & 0 \\ \frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}}\end{bmatrix}$$</center>

정리해보면, 위의 $U$ 행렬은 $$MM^T$$행렬의 eigenvectors들을 이용해 표현했다. 그런데 여기서 사실 중요한 것은 해당 eigenvector들이 orthonormal 하도록 normalize를 꼭 해줘야 한다는 것이다.
> 각 벡터들이 서로 orthogonality를 만족하며 길이는 모두 1이 되는 벡터들로 이뤄진 형태

마찬가지로 $V$행렬을 구해보면, 이번에는 $$MM^T$$가 아닌 $$M^T$$ 행렬의 eigenvector들의 형태로 표현한다. 위의 $M$ 행렬의 $$M^TM$$를 계산해보면 다음과 같다.
<center>$$M^TM = \begin{bmatrix} 2 & 2\sqrt{2} & 0 \\ 2\sqrt{2} & 6 & 2 \\ 0 & 2 & 2 \end{bmatrix}$$</center>
이에 대한 eigenvalues, eigenvectors 들을 위에서와 같은 방식으로 계산하면 다음과 같다.

$$\lambda = 8$$일 때, eigenvector $$\mathbf{y}_1$$은 $$(\frac{1}{\sqrt{6}},\frac{3}{\sqrt{12}} ,\frac{1}{\sqrt{12}})$$ 이다. 마찬가지로, $$\lambda = 2$$일 때, eigenvector $$\mathbf{y}_2$$은 $$(\frac{1}{\sqrt{3}},0 ,-\frac{2}{\sqrt{6}})$$ , $$\lambda = 0$$일 때, eigenvector $$\mathbf{y}_3$$은 $$(\frac{1}{\sqrt{2}},-\frac{1}{2} ,\frac{1}{2})$$ 이다. 이를 종합해서 ***right-singular vectors*** 행렬인 $V$ 를 표현하면 다음과 같다.

<center>$$ V = \begin{bmatrix} \frac{1}{\sqrt{6}} &  \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} \\ \frac{3}{\sqrt{12}} & 0 & -\frac{1}{2} \\ \frac{1}{\sqrt{12}} & -\frac{2}{\sqrt{6}} & \frac{1}{2}\end{bmatrix}$$</center>

우리가 구하고자 한 $$U, V$$행렬은 모두 구했고, $\Sigma$ 행렬은 eigenvalues들을 통해 다음과 같이 간단하게 표현할 수 있다. $$\Sigma = \begin{bmatrix} 2\sqrt{2} & 0 & 0 \\ 0 & \sqrt{2} & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

위에서 모두 구한 $$U, \Sigma, V^T$$ 행렬들의 곱은 $M$과 같다.
<center>$$ \begin{bmatrix} 0 & 1 & 1 \\ \sqrt{2} & 2 & 0 \\ 0 & 1 & 1 \end{bmatrix} =
\begin{bmatrix} \frac{1}{\sqrt{6}} &  -\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} \\ \frac{2}{\sqrt{6}} & \frac{1}{\sqrt{3}} & 0 \\ \frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}}\end{bmatrix} \begin{bmatrix} 2\sqrt{2} & 0 & 0 \\ 0 & \sqrt{2} & 0 \\ 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{6}} &  \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} \\ \frac{3}{\sqrt{12}} & 0 & -\frac{1}{2} \\ \frac{1}{\sqrt{12}} & -\frac{2}{\sqrt{6}} & \frac{1}{2}\end{bmatrix}$$</center>
<center>$$M = U\Sigma V^T$$</center>

> 위에서 언급했던, <br>
  "$$MM^T$$와, $$M^TM$$에서 각각 얻은 고유값들 중에 양수인 값들의 루트 값을 이용해 diagonal한 matrix를 생성했을 때, 해당 행렬은 $$\Sigma$$가 된다."
  이 부분을 상기하자.

## SVD with Python
$\ $Numpy로 간단하게 SVD를 구현해보자!!! 과연 진짜 분해 후 다시 곱한 결과가 원래 매트릭스 값들과 같을지?
```python
import numpy as np

### 4 by 3 행렬을 랜덤하게 선언해주자
m = 4
n = 3

a = np.random.rand(m,n)
```
아래와 같은 4 by 3 크기의 행렬을 생성했다.
<center><img src="https://imgur.com/luJl6H0.png" width="50%" height="50%"></center>
위의 행렬을 `np.linalg.svd()` 함수에 입력해 $$U, s, Vh$$ 세 개의 분해된 행렬을 얻는다. <br>
```python
u, s, vh = np.linalg.svd(a)
```
위에서 얻은 세 개의 분해된 행렬들 중에, $s$는 행렬 형태가 아닌, 고유값 벡터의 `array` 형태이기 때문에, 행렬 형태로 바꿔준다.
```python
min_mn_I = np.identity(max(m, n))

for i in range(0,max(m,n)):
    if i < len(s):
        min_mn_I[i,i] = s[i]
    else:
        min_mn_I[i,i] = 0

S = min_mn_I[:m,:n]
```
결과적으로 아래와 같은 $S$ 매트릭스를 얻는다.
<center><img src="https://imgur.com/OYGGNZ1.png" width="50%" height="50%"></center>
최종적으로, $$U, S, Vh$$ 세개의 행렬들을 곱하게 되면 원래 행렬과 같은 행렬을 얻게 된다. WoW
```python
np.dot(u, np.dot(S,vh))
```
<center><img src="https://imgur.com/QFRKlxZ.png" width="50%" height="50%"></center>

> references:
  https://en.wikipedia.org/wiki/Orthogonal_matrix
  https://en.wikipedia.org/wiki/Singular_value_decomposition
  https://en.wikipedia.org/wiki/Conjugate_transpose
  https://mysite.science.uottawa.ca/phofstra/MAT2342/SVDproblems.pdf
