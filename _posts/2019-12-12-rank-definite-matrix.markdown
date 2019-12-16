---
title: " [선형대수학] positive definite, semi-positive definite"
tags: LinearAlgebra Rank FullRank PositiveDefinite SemiPositiveDefinite
---

# Positive Definite 행렬 & Semi positive Definite 행렬
## 고유값으로 정의되지만..
***<center>A matrix is positive definite if it's symmetric and all its eigenvalues are positive</center>***
> 아주 간단한 정의이다. 행렬이 대칭행렬이고 고유값들이 모두 양수이면 된다고 한다. 하지만 여기서 바로 한가지 걱정이 생겨야 한다.
  아 고유값 저거 귀찮게 언제 다 계산하지?!

모든 eigenvalue를 계산하는 것은 matrix dimension이 증가함에 따라 복잡해진다. 당장 2x2 행렬에서 고유값 계산과 3x3 행렬에서 고유값 계산도 복잡도가 꽤나 차이난다. dimension은 1 씩만 늘었는데.. 따라서 좀 더 효율적이고 덜 귀찮은 방법을 찾아야 한다.<br>
다음의 성질을 사용하자.
***<center>행렬이 갖는 모든 eigenvalue의 부호는 해당 행렬 pivot들의 부호와 같다.</center>***
> 3x3 행렬에서, pivot이 2,-3,3 으로 2개가 양수, 1개가 음수라면, 고유값 또한 2개는 양수이고 1개는 음수라는 성질이다. 해당 성질에 대한 증명은 (어렵다.)

위의 성질을 이용하면 처음 제시된 positive definite 정의를 다음과 같이 바꿀 수 있다.
***<center>A matrix is positive definite if it's symmetric and all its pivots are positive</center>***
> 해당 매트릭스가 symmetric이며, 모든 pivots value가 양수이면 positive definite matrix로 본다.

즉, pivot들의 부호만 확인하면 된다.

아래의 경우를 보면서 pivot을 통한 positive definite 확인을 해보자.
<center>$$\begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$$</center>
위의 행렬을 Gaussian elimination을 통해 변경하면 다음과 같다.
<center>$$\begin{pmatrix} 1 & 2 \\ 0 & -3 \end{pmatrix}$$</center>
행렬의 대각선에 있는 값들은 각각 1, -3 이며 해당 pivot 값들 중 1개는 양수이며 1개는 음수이다. 따라서 eigenvalue 또한 (우리가 계산은 아직 안해봤지만) 1개는 양수이고 1개는 음수임을 알 수 있다.

## pivot 계산도 쉽지가 않은데..?
$\ $k번째 pivot 값은 다음과 같이 쉽게 계산할 수 있다.
<center>$$d_k = \frac{det(A_k)}{det(A_{k-1})}$$</center>
여기에서 $$A_k$$는 upper left k x k submatrix에 해당한다. 다음 범위 $$1 \le k \le n$$ 에 해당하는 모든 $k$에 대하여 다음이 $$det(A_k)$$ 성립한다면 모든 pivot 값들은 양수임이 확인 될 것이다. 따라서 모든 submatrix 의 determinants 값이 양수임을 확인하면 된다. 아래의 행렬이 positive definite일지 계산해보자.
<center>$$\begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}$$</center>
<center>$$d_1 = 2$$</center>
<center>$$d_2 = \begin{vmatrix} 2 & -1 \\ -1 & 2 \end{vmatrix} = 3$$</center>
<center>$$d_3 = \begin{vmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{vmatrix} = 4$$</center>
$$d_1, d_2, d_3 > 0 $$ 이므로, positive definite 행렬임이 확인된다.

> 2x2 행렬 $$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$$ 에서 $$determinant = ad\ -\ bc$$ 로 쉽게 계산할 수 있다.<br>
 마찬가지로, 3x3 행렬 $$\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}$$ 에서 $$determinant = a(ei\ -\ fh) - b(di\ -\ fg) + c(dh\ -\ eg)$$로 상대적으로 쉽게 계산 할 수 있다. 하지만, 행렬 차원수가 커짐에 따라 이러한 방법도 점점 복잡해진다.

## energy-based definition & $$R^TR$$ definition
### energy-based definition of positive definite
$\ $조금 더 행렬스러운 계산 접근으로 positive definite를 정의해보자. $$\mathbf{x} \ne 0$$인 $$\mathbf{x}$$가 $A$의 고유벡터라면, 이 경우에 다음이 성립한다. $$\mathbf{x}^T\mathbf{Ax} = \lambda \mathbf{x}^T \mathbf{x}$$<br>
$\ $여기에서, 만약 $$\lambda > 0$$ 이라면, $$\mathbf{x}^T\mathbf{x} > 0 $$ 이므로, 항상 다음이 성립해야만 한다. $$\mathbf{x}^T\mathbf{Ax} > 0$$.<br>
$\ $즉, 다음과 같이 positive definite 매트릭스에 대한 정의를 이끌어낼 수 있다.
***<center>A matrix is positive definite if $$\mathbf{x}^T\mathbf{Ax} > 0$$ for all vectors $$\mathbf{x}\ \ne 0.$$</center>***
$\ $물리학에서 상태 $$\mathbf{x}$$에 있는 시스템의 **energy** 는 보통 $$\mathbf{x}^T\mathbf{Ax}$$ (또는, $$\frac{1}{2}\mathbf{x}^T\mathbf{Ax}$$)로 자주 표현되기 때문에 이와 같이 positive definite 매트릭스를 정의하는 것을 ***energy-based definition*** 으로 부른다. 이를 활용하여 positive definite 매트릭스에 대한 또다른 정의를 유도해볼 수도 있다.<br>
### $$R^TR$$ definition of positive definite
$\ $ 구성 column들이 서로 독립적이며 직사각형 형태인 매트릭스 $$R$$ 이 있다고 했을 때, $$ A = R^TR $$ 로 작성될 수 있는 모든 매트릭스 $$A$$는 positive definite 매트릭스이다. ***$$ A = R^TR $$ 를 만족하는 $A$는 positive definite 매트릭스이다.*** 라는 이 정의는 energy-based 정의를 이용해 쉽게 증명될 수 있다. 아래의 식을 보자.
<center>$$\mathbf{x}^T\mathbf{Ax} = \mathbf{x}^T\mathbf{R}^T\mathbf{Rx} = (\mathbf{Rx}^T)(\mathbf{Rx}) = \lVert \mathbf{Rx}\rVert^2$$</center>
$\ $만약 $$R$$의 열들이 linearly independent 이고, $$\mathbf{x} \ne 0$$ 이라면, $$\mathbf{Rx} \ne 0$$ 이 성립하며, 따라서 $$\mathbf{x}^T\mathbf{Ax} > 0$$ 을 만족한다. 최종적인 $$\lVert \mathbf{Rx}\rVert^2$$ 값이 양수이므로, 결과적으로 $$\mathbf{x}^T\mathbf{Ax} > 0$$ 을 만족한다. 따라서 다음과 같이 positive definite 매트릭스에 대해 정의할 수 있다.
***<center>A matrix $A$ is positive definite if and only if it can be written as $$A = R^TR$$ for some possibly rectangular matrix $R$ with independent columns</center>***
$\ $마지막으로, 해당 매트릭스의 모든 고유값이 전부 양수가 아니라, 0 이상을 만족할 때에는, ***positive definite***가 아니라 ***positive semidefinite*** 라고 말한다. 다음 매릭스가 ***positive semidefinite*** 를 만족하기 위해서는 $b$의 값이 어떻게 돼야할 지 계산해보자.<br>
<center>$$\begin{pmatrix} 2 & -1 & b \\ -1 & 2 & -1 \\ b & -1 & 2 \end{pmatrix}$$</center>
$\ $위 매트릭스의 determinant가 항상 0 이상이 되도록 하는 $b$의 값을 찾는 문제와 같다.<br>
<center>$$d_3 = \begin{vmatrix} 2 & -1 & b \\ -1 & 2 & -1 \\ b & -1 & 2 \end{vmatrix} $$</center>
<center>$$= 2(4-1) - (-1)(-2+b) + b(1-2b)$$</center>
<center>$$= -2b^2 + 2b + 4 \geq 0 $$</center>
<center>$$\Rightarrow b^2 - b -2 \leq 0$$</center>
<center>$$(b-2)(b+1) \leq 0$$</center>
<center>$$\therefore -1 \leq b \leq 2$$</center>

A (real) symmetric matrix has a complete set of orthogonal eigenvectors for which the corresponding eigenvalues are are all real numbers. For non-symmetric matrices this can fail. For example, a rotation in two dimensional space has no eigenvector or eigenvalues in the real numbers, you must pass to a vector space over the complex numbers to find them.

If the matrix is additionally positive definite, then these eigenvalues are all positive real numbers. This fact is much easier than the first, for if 𝑣 is an eigenvector with unit length, and 𝜆 the corresponding eigenvalue, then

𝜆=𝜆𝑣𝑡𝑣=𝑣𝑡𝐴𝑣>0
where the last equality uses the definition of positive definiteness.

The importance here for intuition is that the eigenvectors and eigenvalues of a linear transformation describe the coordinate system in which the transformation is most easily understood. A linear transformation can be very difficult to understand in a "natural" basis like the standard coordinate system, but each comes with a "preferred" basis of eigenvectors in which the transformation acts as a scaling in all directions. This makes the geometry of the transformation much easier to understand.

For example, the second derivative test for the local extrema of a function 𝑅2→𝑅 is often given as a series of mysterious conditions involving an entry in the second derivative matrix and some determinants. In fact, these conditions simply encode the following geometric observation:

If the matrix of second derivatives is positive definite, you're at a local minimum.
If the matrix of second derivatives is negative definite, you're at a local maximum.
Otherwise, you are at neither, a saddle point.
You can understand this with the geometric reasoning above in an eigenbasis. The first derivative at a critical point vanishes, so the rates of change of the function here are controlled by the second derivative. Now we can reason geometrically

In the first case there are two eigen-directions, and if you move along either the function increases.
In the second, two eigen-directions, and if you move in either the function decreases.
In the last, there are two eigen-directions, but in one of them the function increases, and in the other it decreases.
Since the eigenvectors span the whole space, any other direction is a linear combination of eigen-directions, so the rates of change in those directions are linear combinations of the rates of change in the eigen directions. So in fact, this holds in all directions (this is more or less what it means for a function defined on a higher dimensional space to be differentiable). Now if you draw a little picture in your head, this makes a lot of sense out of something that is quite mysterious in beginner calculus texts.

This applies directly to one of your bullet points
