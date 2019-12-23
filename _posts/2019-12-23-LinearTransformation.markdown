---
title: " [선형대수학] Linear Transformation (선형변환)"
tags: LinearAlgebra Transformation Basis Span Vector
---

# Basis Vectors, Span
$\ $ 고등학교에서 행렬과 행렬 간의 곱을 구하는 방법은 우리가 이미 배워서 익히 안다. 또한, 단위 벡터를 활용해서 특정 공간에서 특정 벡터를 단위 벡터만을 활용해서 표현할 수 있음도 알고 있다. 벡터의 개념과 행렬 표현식을 조합해서 사용하는 법에 대해 알아보자.
$\ $ $$x, y$$ 평면에서 $x$축의 양의 방향으로 1만큼 움직이는 벡터를 $i$, $y$축의 양의 방향으로 1만큼 움직이는 벡터를 $j$라고 놓고 이 두개의 $i$, $j$벡터를 단위벡터라고 말하기로 헸었다. 단위벡터는 영어로 basis vector라고 하기로 한다. 이를 행렬로 표현하면 다음과 같다.
<center> $$ i = \begin{bmatrix} 1 \\ 0 \end{bmatrix} $$</center>
<center> $$ j = \begin{bmatrix} 0 \\ 1 \end{bmatrix} $$</center>

x, y 평면에서 어떠한 점이든지 우리는 $i, j$ 벡터들의 조합으로 도달할 수 있다. 예를 들어, $$(10, -5)$$에 해당하는 점은 $$10 \begin{bmatrix} 1 \\ 0 \end{bmatrix} - 5 \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$ 와 같이 표현할 수 있다. 이렇게 basis vector와 같은 벡터들을 선형적으로 조합하여(Linear Combination), 다양한 점(목적지)로 표현할 수 있는데 이런 선형조합의 결과물을 Span이라고 부르도록 한다. 임의의 실수 $$a, b$$와 basis vector들을 선형적으로 조합해서 얻을 수 있는 다음과 같은 결과물은, $$x, y$$ 평면의 어디든지 도달할 수 있다.
<center>$$ a\begin{bmatrix} 1 \\ 0 \end{bmatrix} + b\begin{bmatrix} 0 \\ 1 \end{bmatrix}</center>
물론, 만약 basis vector가 서로 linearly dependent 하다면, 그 두개의 벡터로는 $$x, y$$ 평먼의 모든 곳에 도달할 수 없는 경우에 해당한다. 예를 들어, $$\begin{bmatrix} 1 \\ 1 \end{bmatrix}$$ 를 $m$ 단위벡터로 잡고, $$\begin{bmatrix} -3 \\ -3 \end{bmatrix}$$ 을 $n$ 단위벡터로 잡게 되면, 두개의 벡터들이 서로 선형적으로 종속이므로 모든 공간을 이 두개의 단위벡터의 선형조합으로 표현할 수 없다.
> 조금 더 기하학적으로 생각해보자. 위에 보여진 $m$ 단위벡터는, $$x, y$$ 평면에 표현해보면, (0,0)을 원점으로 하면서 (1,1)을 가리키는 화살표에 해당한다. 마찬가지로 $n$ 단위벡터는 $$x, y$$ 평면에서 (0,0)을 원점으로 하면서, (-3,-3)을 가리키는 단위벡터이다. 두 개의 벡터를 떠올려 보면 완전히 방향만 반대일 뿐, 하나의 직선에 최종점 두개가 놓이게 되는데, 이런 경우 두개의 벡터를 활용해서 해당 직선 바깥에 있는 점은 선형조합으로 표현할 수 없게 된다.



> reference :
  https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3
