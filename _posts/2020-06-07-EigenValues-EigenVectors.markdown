---
title: " [선형대수학] Spectral theorem"
tags: LinearAlgebra Eigenvalues Eigenvectors
---

# EigenVectors and Eigenvalues
$$\mathbf{Ax}$$ 형태로 $$\mathbf{A}$ 매트릭스에 $$\mathbf{x}$$를 곱하면 때떄로 운좋게 $$\mathbf{x}$$ 의 스칼라를 곱한, 방향은 그대로인 벡터 형태가 된다. 여기서, $$\mathbf{A}$$는  $n$ by $n$ 매트릭스이다. 이때, 람다는 eigenvalue, $$\mathbf{x}$$는 eigenvector로 부르기로 한다. 그렇다면, 이 아이젠 형제들을 우리가 사용함으로써 도대체 무엇이 좋은걸까? 이는 $$\mathbf{A}^2$$ 매트릭스를 볼 때 알 수 있다. 이 $$\mathbf{A}^2$$ 는 여전히 $n$ by $n$ 행렬이다.
<center>$$\mathbf{Ax}_{i} = \lambda_{i}\mathbf{x}_{i}, i = 1,2,\cdots,n$$</center><br>
> 총 n개의 eigenvector, eigenvalue set이 존재할 수도 있지만, 더 적은 수가 있는 경우도 있다.

<center>양변에 $$\mathbf{A}$$를 곱하면 다음과 같다.</center>
<center>$$\mathbf{A}^2\mathbf{x} = \mathbf{A}(\mathbf{Ax}) = \mathbf{A}(\lambda\mathbf{x}) = \lambda(\mathbf{Ax}) = \lambda(\lambda\mathbf{x}) = \lambda^{2}\mathbf{x}$$</center><br>
$$\mathbf{x}$$는 $$\mathbf{A}$$뿐 아니라, $$\mathbf{A}^2$$ 의 eigenvector 이기도 하다는 결론을 얻는다. 또한, eigenvalue 는 $$\lambda^2$$가 된다.<br>
$$\mathbf{A}^{k}\mathbf{x} = \lambda^{k}\mathbf{x}$$ 의 경우에서 모든 $k$에 대하여 이 성질은 성립하며, $$ k = -1 $$이라면, $$\mathbf{A}^{-1}\mathbf{x} = \frac{1}{\lambda}\mathbf{x}$$가 된다. 심지어 다음과 같이 응용할 수도 있다. growth population에 일반적으로 쓰이는 exponential case의 식을 아래와 같이 작성할 수 있다. $$e^{\mathbf{A}t} = e^{\lambda t}\mathbf{x}$$
> special is GOOD! but useful is BETTER! 더욱 유용한 예시들을 확인해보자.

## Combination of EigenVectors
$\ $임의의 벡터 $$\mathbf{v}$$를 잡자. $$\mathbf{A}$$ 행렬에 총 $n$개의 독립적인 eigenvector들도 존재하는 상황으로 가정해보자. 이 독립적인 eigenvector들을 각각 basis로 잡고, $$\mathbf{v}$$가 eigenvector들 중 하나라면, $$\mathbf{v}$$는 이 basis들의 combination으로 표현할 수 있다.<br>
<center>$$\mathbf{v} = c_{1}x_{1} + c_{2}x_{2} + \cdots + c_{n}x_{n}$$</center><br>
<center>\mathbf{A}^{k}\mathbf{v} = c_{1}\lambda_{1}^{k}x_{1} + \cdots + c_{n}\lambda_{n}^{k}x_{n}</center>
이를 잘 활용하면, $$\mathbf{A}^{k}$$를 빠르게 계산할 수 있다.<br>
