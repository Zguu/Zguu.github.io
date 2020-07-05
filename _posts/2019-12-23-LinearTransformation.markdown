---
title: " [선형대수학] Linear Transformation (선형변환)"
tags: LinearAlgebra
---
# Linear Transformation
$\ $고등학교에서 행렬과 행렬 간의 곱을 구하는 방법은 우리가 이미 배워서 익히 안다. 또한, 단위 벡터를 활용해서 특정 공간에서 특정 벡터를 단위 벡터만을 활용해서 표현할 수 있음도 알고 있다. 벡터의 개념과 행렬 표현식을 조합해서 사용하는 법에 대해 알아보자.
$\ $행렬들의 곱을 계산하는 방법은 이미 우리가 알고 있지만, 해당 연산이 의미하는 바가 기하학적으로 이해하기 위해서는 Linear Transformation을 이해할 필요가 있다. 각 행렬의 column space을 basis vector로 이해할 때, 이 벡터들의 선형적 조합의 결과물은 해당 공간에서 Linear Transformation에 해당한다. <br>
이 말이 어렵게 느껴진다면 아래의 개념들을 차근차근 복습하면서, 왜 행렬의 곱이 Linear Transformation에 해당하는 지에 대한 통찰을 얻도록 하자.

## Basis Vectors, Span
$\ $ $$x, y$$ 평면에서 $x$축의 양의 방향으로 1만큼 움직이는 벡터를 $i$, $y$축의 양의 방향으로 1만큼 움직이는 벡터를 $j$라고 놓고 이 두개의 $i$, $j$벡터를 단위벡터라고 말하기로 했었다. 단위벡터는 영어로 basis vector라고 하기로 한다. 이를 행렬로 표현하면 다음과 같다.
<center> $$ i = \begin{bmatrix} 1 \\ 0 \end{bmatrix} $$</center>
<center> $$ j = \begin{bmatrix} 0 \\ 1 \end{bmatrix} $$</center>

$$x, y$$ 평면에서 어떠한 점이든지 우리는 $i, j$ 벡터들의 조합으로 도달할 수 있다. 예를 들어, $$(10, -5)$$에 해당하는 점은 $$10 \begin{bmatrix} 1 \\ 0 \end{bmatrix} - 5 \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$ 와 같이 표현할 수 있다. 이렇게 basis vector와 같은 벡터들을 선형적으로 조합하여(Linear Combination), 다양한 점(목적지)로 표현할 수 있는데 이런 선형조합의 결과물을 Span이라고 부르도록 한다.
> Span의 개념과 필수적으로 함꼐 이해돼야 하는 것이 Rank, Full Rank와 같은 것들인데, 해당 개념에 대해서는 나중에 추가적으로 적자. 갈길이 멀다. (좀 이따가 기차 내려야됨)

임의의 실수 $$a, b$$와 basis vector들을 선형적으로 조합해서 얻을 수 있는 다음과 같은 결과물은, $$x, y$$ 평면의 어디든지 도달할 수 있다.
<center>$$ a\begin{bmatrix} 1 \\ 0 \end{bmatrix} + b\begin{bmatrix} 0 \\ 1 \end{bmatrix}$$</center>
물론, 만약 basis vector가 서로 linearly dependent 하다면, 그 두개의 벡터로는 $$x, y$$ 평면의 모든 곳에 도달할 수 없는 경우에 해당한다. 예를 들어, $$\begin{bmatrix} 1 \\ 1 \end{bmatrix}$$ 를 $m$ 단위벡터로 잡고, $$\begin{bmatrix} -3 \\ -3 \end{bmatrix}$$ 을 $n$ 단위벡터로 잡게 되면, 두개의 벡터들이 서로 선형적으로 종속이므로 $$x, y$$ 평면 공간에 있는 모든 점들을 이 두개의 단위벡터의 선형조합만으로는 표현할 수 없다.
> 조금 더 기하학적으로 생각해보자. 위에 보여진 $m$ 단위벡터는, $$x, y$$ 평면에 표현해보면, (0,0)을 원점으로 하면서 (1,1)을 가리키는 화살표에 해당한다. 마찬가지로 $n$ 단위벡터는 $$x, y$$ 평면에서 (0,0)을 원점으로 하면서, (-3,-3)을 가리키는 단위벡터이다. 두 개의 벡터를 떠올려 보면 완전히 방향만 반대일 뿐, 하나의 직선에 최종점 두개가 놓이게 되는데, 이런 경우 두개의 벡터를 활용해서 해당 직선 바깥에 있는 점은 선형조합으로 표현할 수 없게 된다.

## Basis Change
$\ $우리가 항상 그래왔고 위에서도 그랬듯이, 항상 basis vector를 $$ i = \begin{bmatrix} 1 \\ 0 \end{bmatrix} $$, $$j = \begin{bmatrix} 0 \\ 1 \end{bmatrix} $$ 와 같이 설정해야만 할까? 라고 묻는다면 당연히 다른 방법이 존재할 것이다. 바로 예시를 보자.
<center>$$\begin{bmatrix} 1 & 3 \\ -2 & -1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ -5 \end{bmatrix}$$</center>
위의 결과는 다음과 같이 해석될 수 있다.
<center>$$ 2 \begin{bmatrix} 1 \\ -2 \end{bmatrix}  + \begin{bmatrix} 3 \\ -1 \end{bmatrix} = \begin{bmatrix} 5 \\ -5 \end{bmatrix}$$</center>
위의 예시에서, 두개의 basis vector는 각각 $$m = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$와 $$n = \begin{bmatrix} 3 \\ -1 \end{bmatrix}$$로 표현됐는데, 이렇게 basis vector를 우리가 기존에 알고있던 길이 1의 식상한 벡터들과 다르게 잡는 것이 어떤 의미가 있는지 기하학적으로 이해할 필요가 있다. $m$ 벡터는 $x$축으로 1만큼, $y$축으로 -2만큼 방향으로 가리키는 화살표(벡터)를 의미한다. $n$벡터는 $x$축으로 3만큼, $y$축으로 -1만큼 방향으로 가리키는 화살표(벡터)를 의미한다. 우변에 결과로 나오는 $$\begin{bmatrix} 5 \\ -5 \end{bmatrix}$$는 두 벡터의 선형적 조합의 결과이며, 기하학적으로 이해해봤을 때 $$m$$벡터가 가리키는 방향으로 2번, $$n$$ 벡터가 가리키는 방향으로 한 번 이동한 결과이다. 이는 기저에 깔려있는 $$x, y$$ 평면의 기본 단위를 바꾼다고 이해할 수 있다.
정리해보면, 좌변에 있는 $$\begin{bmatrix} 1 & 3 \\ -2 & -1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$가 의미하는 바는, $$\begin{bmatrix} 1 & 3 \\ -2 & -1 \end{bmatrix}$$의 column space들을 basis vector로 하는 $x, y$평면에서 각각 2만큼 1만큼 이동한 결과라는 것이다. 이것이 사실 우리가 잘 알고있는 행렬간의 곱에 대한 기하학적 이해로 볼 수 있다.
> 이미지로 표현해야 하는데.... 현재 이미지 업로드할 여유가 없다. 나중에 추가하도록 한다.

> reference :
  https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3
