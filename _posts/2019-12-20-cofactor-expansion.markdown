---
title: " [선형대수학] Cofactor expansion"
tags: LinearAlgebra Determinant Cofactor Expansion
---

# Determinant
$\ $ $$2 \times 2$$ 행렬에 있어서, 해당 행렬의 determinant가 0이 아닐 때에 해당 행렬의 역행렬이 존재한다.
이러한 determinant와 역행렬 존재유무의 관계는 더 큰 사이즈의 행렬에도 당연히 성립한다. $$\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$ 행렬의 determinant는 $$ ad - bc $$로 쉽게 계산할 수 있음을 알고 있다. 하지만, 행렬의 사이즈가 커감에 따라 determinant 계산이 점점 쉽지 않음을 알 수 있는데, 따라서 다양한 determinant 방법들이 존재하며 그에 대해 알아보자.
## Cofactor Expansion
$\ $우선, $$3\times 3$$ 행렬의 determinant를 구하기 위해 다음과 같은 방법을 사용할 수 있다.
<center> $$det(A) = a_{11}det(A_{11}) - a_{12}det(A_{12}) + \cdots + (-1)^{(1+n)}a_{1n}det(A_{1n})$$</center>
<center>$$ = \sum_{j=1}^n (-1)^{1+j}a_{1j}det(A_{1j})$$</center>
$ \$위의 determinant 공식을 아래에 적용해보자.<br>
<center>$$A = \begin{bmatrix} 1 & 5 & 0 \\ 2 & 4  & -1 \\ 0 & -2 & 0 \end{bmatrix}$$ </center>
$$det(A) = a_{11}det(A_{11}) - a_{12}det(A_{12}) + a_{13}det(A_{13})$$ 에 대한 계산을 진행한다.
<center>
$$det(A) = 1\cdot det\begin{bmatrix} 4 & -1 \\ -2 & 0 \end{bmatrix} - 5\cdot det\begin{bmatrix} 2 & -1 \\ 0 & 0 \end{bmatrix} + 0\cdot\begin{bmatrix} 2 & 4 \\ 0 & -2 \end{bmatrix}$$</center>
<center>$$ = 1(0 - 2) - 5(0 - 0) + 0(-4 - 0) = -2 $$</center>
위와 같이 determinant 계산을 진행한다. 이를 일반화시켜서, A 행렬의 ***(i, j)-cofactor*** 를 $$C_{ij}$$로 부르기로 하며, 수식으로는 아래와 같다.
<center>$$C_{ij} = (-1)^{i+j}det(A_{ij})$$</center>
<center> $$det(A) = a_{11}C_{11} + a_{12}C_{12} + \cdots + a_{1n}C_{1n}$$</center>
$\ $이 식을 행렬 A의 ***cofactor expansion across the first row*** 라고 부르자. across the first row라는 말에서 눈치 챌 수 있겠지만, 해당 방법은 첫번째 행을 축으로 잡지 않고, 다른 행이나 열을 축으로 잡고 진행해나갈 수도 있다.
$i$번째 행을 축으로 진행하는 cofactor expansion은 아래와 같이 쓸 수 있다.
<center> $$det(A) = a_{i1}C_{i1} + a_{i2}C_{i2} + \cdots + a_{in}C_{in}$$</center>
$j$번째 열을 축으로 진행하는 cofactor expansion은 아래와 같이 쓸 수 있다.
<center> $$det(A) = a_{1j}C_{1j} + a_{2j}C_{2j} + \cdots + a_{3j}C_{3j}$$</center>

다음의 행렬을 cofactor expansion across the ***third row*** 방식으로 표현하고 계산해보자.
<center>$$A = \begin{bmatrix} 1 & 5 & 0 \\ 2 & 4  & -1 \\ 0 & -2 & 0 \end{bmatrix}$$ </center>
<center>$$det(A) = a_{31}C_{31} + a_{32}C_{32} + a_{33}C_{33}$$</center>
<center>$$ = (-1)^{3+1}a_{31}det(A_{31}) + (-1)^{3+2}a_{32}det(A_{32}) + (-1)^{3+3}a_{33}det(A_{33})$$</center>
<center>$$ = 0\begin{vmatrix} 5 & 0 \\ 4 & -1\end{vmatrix} - (-2)\begin{vmatrix} 1 & 0 \\ 2 & -1\end{vmatrix} + 0\begin{vmatrix} 1 & 5 \\ 2 & 4\end{vmatrix}$$</center>
<center>$$ = 0 + 2(-1) + 0 = -2$$<center>
