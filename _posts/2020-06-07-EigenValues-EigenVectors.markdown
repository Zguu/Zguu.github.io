---
title: " [선형대수학] Application of Eigenvectors and Eigenvalues"
tags: LinearAlgebra Eigenvalues Eigenvectors
---

# EigenVectors and Eigenvalues

$$\mathbf{Ax}$$ 형태로 $$\mathbf{A}$$ 매트릭스에 $$\mathbf{x}$$를 곱하면 때떄로 운좋게 $$\mathbf{x}$$ 의 스칼라를 곱한, 방향은 그대로인 벡터 형태가 된다. 여기서, $$\mathbf{A}$$는  $n$ by $n$ 매트릭스이다. 이때, 람다는 eigenvalue, $$\mathbf{x}$$는 eigenvector로 부르기로 한다. 그렇다면, 이 아이젠 형제들을 우리가 사용함으로써 도대체 무엇이 좋은걸까? 이는 $$\mathbf{A}^2$$ 매트릭스를 볼 때 알 수 있다. 이 $$\mathbf{A}^2$$ 는 여전히 $n$ by $n$ 행렬이다.
<center>$$\mathbf{Ax}_{i} = \lambda_{i}\mathbf{x}_{i}, \  i = 1,2,\cdots,n$$</center><br>
> 총 n개의 eigenvector, eigenvalue set이 존재할 수도 있지만, 더 적은 수가 있는 경우도 있다.

<center>양변에 $\mathbf{A}$를 곱하면 다음과 같다.</center>
<center>$$\mathbf{A}^2\mathbf{x} = \mathbf{A}(\mathbf{Ax}) = \mathbf{A}(\lambda\mathbf{x}) = \lambda(\mathbf{Ax}) = \lambda(\lambda\mathbf{x}) = \lambda^{2}\mathbf{x}$$</center><br>
$$\mathbf{x}$$는 $$\mathbf{A}$$뿐 아니라, $$\mathbf{A}^2$$ 의 eigenvector 이기도 하다는 결론을 얻는다. 또한, eigenvalue 는 $$\lambda^2$$가 된다.<br>
$$\mathbf{A}^{k}\mathbf{x} = \lambda^{k}\mathbf{x}$$ 의 경우에서 모든 $k$에 대하여 이 성질은 성립하며, $$ k = -1 $$이라면, $$\mathbf{A}^{-1}\mathbf{x} = \frac{1}{\lambda}\mathbf{x}$$가 된다. 심지어 다음과 같이 응용할 수도 있다. growth population에 일반적으로 쓰이는 exponential case의 식을 아래와 같이 작성할 수 있다. $$e^{\mathbf{A}t} = e^{\lambda t}\mathbf{x}$$
> special is GOOD! but useful is BETTER! 더욱 유용한 예시들을 확인해보자.

## Combination of EigenVectors
$\ $임의의 벡터 $$\mathbf{v}$$를 잡자. $$\mathbf{A}$$ 행렬에 총 $n$개의 독립적인 eigenvector들도 존재하는 상황으로 가정해보자. 이 독립적인 eigenvector들을 각각 basis로 잡고, $$\mathbf{v}$$가 eigenvector들 중 하나라면, $$\mathbf{v}$$는 이 basis들의 combination으로 표현할 수 있다.<br>
<center>$$\mathbf{v} = c_{1}x_{1} + c_{2}x_{2} + \cdots + c_{n}x_{n}$$</center><br>
<center>$$\mathbf{A}^{k}\mathbf{v} = c_{1}\lambda_{1}^{k}x_{1} + \cdots + c_{n}\lambda_{n}^{k}x_{n} \cdots (1)$$</center>
이를 잘 활용하면, $$\mathbf{A}^{k}$$를 빠르게 계산할 수 있다.<br>
만약 위의 $$\mathbf{A}^{k}\mathbf{v} = \mathbf{V}_k$$ 로 놓으면, $$\mathbf{V}_{k+1} = \mathbf{A}^{k+1}\mathbf{v} = \mathbf{A}\mathbf{v}_k$$가 된다. 부르기 편하게 이를 discrete case로 임시로 부르자.
> one tep difference equation

만약, 우리가 exponential case 인 $$e^{At} = e^{\lambda t}\mathbf{x}$$를 사용하면, $$dv/dt = \mathbf{A}\mathbf{v}$$가 된다. 부르기 편하게 이를 continuous evolution case로 임시로 부르자.
위에서 살펴본 discrete case, continuous case 모두 $$(1)$$ 식을 활용하여 빠르게 $$A^k$$를 계산할 수 있다.

## Similar Matrices
$$\mathbf{A}$$와 similar(유사)한 행렬 $$\mathbf{B}$$. 여기서 similar의 정의가 무엇일까? 이는 벡터들간의 유사도를 측정하는 것과는 조금 정의가 다른데, 간단히 $$\mathbf{A}$$와 $$\mathbf{B}$$가 같은 eigenvalue를 갖고있다면, 서로 similar한 행렬로 부른다. 수식으로 정의는 아래와 같다.
<center>$$\mathbf{B} = \mathbf{M}^{-1}\mathbf{A}\mathbf{M}$$ 의 관계가 성립하면 서로 similar matrices라고 부른다. $$\mathbf{A}와 \mathbf{B}$$는 또한, 같은 eigenvalues를 갖는다.</center>
즉, "두 행렬이 similar matrices 관계이다." 라는 명제와 "두 행렬이 same eigenvalue를 갖는다." 는 같은 뜻으로 이해할 수 있다. 하지만, $$\mathbf{M}^{-1}\mathbf{A}\mathbf{M}y = \lambda y$$ 관계에서 아래의 두 사실을 혼동해선 안된다.<br>
- $y$는 $$\mathbf{A}$$와 같은 eigenvector를 갖는다. (False)
- $y$는 $$\mathbf{A}$$와 같은 eigenvalue를 갖는다. (True)

<center>$$\mathbf{AM}y = \lambda\mathbf{M}y \to \mathbf{A}(\mathbf{M}y) = \lambda(\mathbf{M}y) $$</center>
즉, eigenvalue는 위에서 보이듯이 $\lambda$ 값으로 변하지 않는다. 하지만, eigenvector는 $$\mathbf{x}$$에서 $$\mathbf{M}y$$ 로 변했다.

## AB and BA is similar
$$\mathbf{AB}$$와 $$\mathbf{BA}$$는 서로 같은 eigenvalue를 갖는다. 이를 간단히 증명하려면, 아래와 같이 둘이 similar함을 보이면 된다. <br>
<center>$$\mathbf{M}(\mathbf{AB})\mathbf{M}^{-1} = \mathbf{BA}$$</center>
이를 만족하는 $$\mathbf{M}$$은 어떤 값일까? 잠시 생각해보면 $$\mathbf{M} = \mathbf{B}$$를 만족하면 된다는 걸 알수 있다.
> $\because \mathbf{BAB}\mathbf{B}^{-1} = \mathbf{BA}$

## $$\sum(\lambda s)$$ & $$\prod(\lambda s)$$
eigenvalue들의 합 또는 곱에 대한 다음과 같은 공식이 존재한다. 아래 공식은 모든 $n$ by $n$ 행렬에 통용된다.
- sum of $\lambda$s  = sum of diagonals  = trace
- multiply of $\lambda$s  = determiant($\mathbf{A}$)

간단한 예시를 들어보면 다음과 같다. $$\mathbf{A} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$ 라는 anti-symmetric 행렬을 보자.
> anti-symmetric 행렬 : tranpose($$\mathbf{A}$) = -\mathbf{A}$$ 를 만족하는 행렬 $\mathbf{A}$

이 행렬은 임의의 행렬 $$\mathbf{x}$$와 곱해졌을 때 해당 행렬 $\mathbf{x}$를 90도 회전시키는 행렬이기도 하다. 기하학적으로 생각해봤을 때, 이 회전행렬 $$\mathbf{A}$$는 자신과 곱해지는 행렬을 90도로 회전시켜버리기 때문에, 원래 행렬을 상수배 했을 때 이 회전된 행렬과 같은 방향을 가르키는 것을 불가능하다. 즉, 이 anti-symmetric 행렬은 eigenvector를 가질 수 없다. 하지만 이 행렬의 eigenvector를 한 번 구해보자.
<center>$$\mathbf{Ax} = \lambda\mathbf{x}</center>
<center>$$\mathbf{Ax} = \lambda\mathbf{x}</center>
