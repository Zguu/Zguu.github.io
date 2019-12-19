---
title: " [선형대수학] LU, LDU decomposition"
tags: LinearAlgebra LU LDU Decomposition GaussianElimination
---

# LU, LDU decomposition
$ \$ Matrix Decomposition의 기본적 형태인 LU, LDU decomposition에 대해서 정리해보자. Decomposition은 Factorization으로 불리기도 하는데, 특정 매트릭스를 2개 이상의 매트릭스로 분리하는 것이다. 예를 들어, $$ m \times m $$ 행렬을, $$ m \times n $$ 행렬과 $$ n \times m$$ 행렬 총 두개의 행렬로 분해하는 것이다. 다항식에서 우리가 인수분해를 하는 것과 비슷하며, 우리가 다항식의 인수분해서 얻을 수 있는 이점과 비슷한 이점을 매트릭스 분해에서도 얻을 수 있다. 해당 매트릭스와 연관된 linear system의 해를 구하거나, 다른 매트릭스와의 곱에서 연산 복잡도를 줄여줄 수 있다. 가장 기본적인 형태의 매트릭스 분해인 LU에 대해서 짚고 넘어가기 전에, Gaussian Elimination(GE)을 상기시킬 필요가 있다. GE은 간단하기도 하면서도, 실제 연산을 진행할 땐 성가신 계산 과정이지만 이 반복작업에서 교훈을 얻어야할 점이 분명히 있으므로 짚고넘어가도록 한다.
## Gaussian Elimination = Matrix Multiplication
$\ $Gaussian Elimination(GE)는 Row Reduction이라고도 부르는데, 우리가 중고등학교에서 배웠던 연립방정식의 가감법을 행렬 형태에 그대로 적용하며 연산을 진행하는 것이다. GE는 어렵지 않은 개념이므로, 생략하도록 한다. 아래의 간단한 GE 프로세스만 보더라도 잊는 내용을 충분히 상기시킬 수 있다. <br>
여기서 중요한 것은 행렬의 GE는 사실 행렬 간의 곱으로 표현된다는 것이며, 이에 대해 우리가 익숙해질 필요가 있다.$$2 \times 2$$ 행렬 예시를 보자.
<center>$$\begin{bmatrix} 4 & 5 \\ 8 & 11 \end{bmatrix}$$</center>
위의 매트릭스의 첫번째 행에 2를 곱한 후 해당 값들을 두번째 행에서 빼주면 다음과 같은 결과를 얻는다.
<center>$$\rightarrow\begin{bmatrix} 4 & 5 \\ 0 & 1 \end{bmatrix}$$</center>
사실 여기에서 각 행의 pivot들을 1로 만들도록 해주는 과정까지 진행해야 완벽한 GE라고 볼 수 있지만, 우리는 우선 이렇게 upper triangular matrix를 완성하고 멈추도록 해보자.
<center>$$A = \begin{bmatrix} 4 & 5 \\ 8 & 11 \end{bmatrix}$$</center>
<center>$$\rightarrow \begin{bmatrix} 4 & 5 \\ 0 & 1 \end{bmatrix} = A^{\prime}$$</center>
처음 행렬 형태를 $A$ 행렬, GE 과정을 마친 결과 행렬을 $$A^{\prime}$$ 행렬로 지정하고, 해당 행렬들이 서로 행렬의 곱으로 표현될 수 있음을 아래를 통해 확인해보자.
<center>$$\begin{bmatrix} 1 & 0 \\ -2 & 1 \end{bmatrix}\begin{bmatrix} 4 & 5 \\ 8 & 11 \end{bmatrix} = \begin{bmatrix} 4 & 5 \\ 0 & 1 \end{bmatrix}$$</center>
<center>$$LA = A^{\prime}$$</center>
원래의 $A$ 행렬 좌측에 $$L = \begin{bmatrix} 1 & 0 \\ -2 & 1\end{bmatrix}$$ 행렬을 곱함으로써, $A^{\prime}$행렬을 얻었다. 사실 여기서 $L$ 행렬(left matrix)의 원소 -2 가 그냥 0 이었다면 $L$ 행렬은 ***identity*** 행렬에 불과하다. 하지만 2번째 행 첫번째 원소를 -2 로 바꾼 행렬 $L$를 곱함으로써 GE를 진행한 것과 같은 결과를 내게 됐는데, 아직까지 이에 대한 인싸이트가 와닿지 않는다면 하나의 예시를 더 보도록 하자.
<center>$$B = \begin{bmatrix} 10 & -4 \\ 5 & 8 \end{bmatrix}$$</center>
<center>$$\rightarrow \begin{bmatrix} 10 & -4 \\ 0 & 10 \end{bmatrix} = B^{\prime}$$</center>
마찬가지로, GE를 통해 $B$행렬을 upper triangular matrix 형태로 바꿨다. 이것을 다시 행렬의 곱으로 표현하면 다음과 같다.
<center>$$\begin{bmatrix} 1 & 0 \\ -\frac{1}{2} & 1 \end{bmatrix}\begin{bmatrix} 10 & -4 \\ 5 & 8 \end{bmatrix} = \begin{bmatrix} 10 & -4 \\ 0 & 10 \end{bmatrix}$$</center>
<center>$$LB = B^{\prime}$$</center>
좌측에 있는 $$\begin{bmatrix} 1 & 0 \\ -\frac{1}{2} & 1 \end{bmatrix}$$ 행렬이 역시 $I$ 행렬의 원소 하나가 $$-\frac{1}{2}$$로 변경된 것이다. 이쯤되면 우리가 왜 이 지겨운 GE를 행렬의 곱 형태로 다시 바라보고 있는 지에 눈치를 챌 수 있을 것이다.<br>
$\ $ 우리가 GE 과정에서 행렬들을 triangular 형태로 만들기 위해, (다항식에서는 우리가 가감법을 하기 위해 한 행에 특정 상수를 곱하고, 일부 미지수들의 계수 값이 같도록 하는 것) 특정 행렬에 곱했던 상수가, left matrix 의 해당 위치에 다시 나타난다는 것이다. 우리가 여기에서 얻을 수 있는 교훈은 총 두가지다.

- GE 과정에서 결과 행렬은 일반적으로 upper triangular matrix이며, left matrix는 lower triangular matrix의 형태를 보인다.
- lower triangular matrix 에 해당하는 왼쪽 행렬은 ***identity matrix*** ($I$)의 일부 변형된 꼴이며, 우리가 GE 진행 과정에서 각 행의 가감을 위해 곱하거나 나눴던 상수들이 해당 위치 원소들을 대체한다. 지금까지 확인한 GE와 multiplication 간의 관계를 일반화해서 얘기해보면, 결국 GE는 큰 범위에서 matrix multiplication의 일종이라는 것이다. 물론, 위에서 예를 든 행렬들 외에 다른 변칙적 행렬들 같은 경우는 permutation 행렬 ($P$)를 적절한 위치에 곱해줘야하는 필요도 있으나, 그 또한 행렬의 곱 연산 범위 내에서 모두 표현이 가능하다.
최종적으로, 어떤 행렬 $$X$$ 는 $$LX = U$$로 표현될 수 있다는 점을 꼭 명심하고 넘어가자.<br>
***여기까지만 오면 사실 LU decomposition은 거의 다 된거나 다름없다.***
> L = lower triangular matrix, U = upper triangular matrix

## LU decomposition
$\ $위에서 우리는 어떤 행렬 $X$는 $$LX = U$$ 형태로 표현될 수 있다는 점에 배웠다. 여기에서 $L$ 함수의 inverse matrix만 간단히 좌변과 우변의 좌측 항에 곱해주게 되면 $$ X = L^{-1}U $$ 형태를 얻는다. $$L^{-1}$$ 은 또 다시 lower triangular matrix 형태를 띄게 되므로 결과적으로는, $$ X = LU $$ 형태를 보여준다. 다시 예를 들어보자. <br>
<center>$$X = \begin{bmatrix} 3 & 4 \\ 6 & 5 \end{bmatrix}$$</center>
<center>$$\begin{bmatrix} 1 & 0 \\ -2 & 1 \end{bmatrix}\begin{bmatrix} 3 & 4 \\ 6 & 5 \end{bmatrix} = \begin{bmatrix} 3 & 4 \\ 0 & -3 \end{bmatrix}$$</center>
위와 같이 GE 결과를 $$LX = X^{\prime}$$ 형태로 표현한 후, $L$행렬의 역함수를 구해서 양변의 좌측에 곱해준다. $$\begin{bmatrix} 1 & 0 \\ -2 & 1 \end{bmatrix}$$ 행렬의 역함수는 $$\begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$$이며, 해당 역함수를 양 변에 곱한 형태는 다음과 같다.
<center>$$\begin{bmatrix} 3 & 4 \\ 6 & 5 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} 3 & 4 \\ 0 & -3 \end{bmatrix}$$</center>
즉, 원래 행렬 $X$가 완벽하게 lower triangular matrix인 $L$과, upper triangular matrix인 $U$ 두 행렬의 곱으로 표현됐다. 우리가 목표로 했던 Matrix decompoistion (분햬)를 완성했다. 물론, 위의 GE 과정에서 pivot들의 위치에 따라 permutation matrix 를 곱해서 진행해야하는 경우도 있지만, 우리는 위와 같은 간단한 예시만을 살펴보았다. 추가적으로 해당 LU decompoistion 이 $$3 \times 3$$ 행렬과 같은 더 큰 사이즈의 matrix에도 적용이 되는지 아래를 통해 살펴보자.

<center>$$\begin{bmatrix} 3 & 2 & 1 \\ 6 & 2 & 5 \\ 9 & 1 & 8 \end{bmatrix}$$</center>
아래와 같이 row reduction을 한 번 진행한 GE 중간 결과 행렬을 우측 항에, 해당 결과 행렬을 만들기 위한 행렬 곱 형태를 좌측 항에 적어준다.
<center>$$\begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} 3 & 2 & 1 \\ 6 & 2 & 5 \\ 9 & 1 & 8 \end{bmatrix} = \begin{bmatrix} 3 & 2 & 1 \\ 0 & -2 & 3 \\ 9 & 1 & 8 \end{bmatrix}$$</center>

위의 세 개 행렬에서 우측 항에 있는 GE 중간 결과 행렬에 대해 추가적으로 GE를 계속 진행한다.

<center>$$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -3 & 0 & 1 \end{bmatrix}\begin{bmatrix} 3 & 2 & 1 \\ 0 & -2 & 3 \\ 9 & 1 & 8 \end{bmatrix} = \begin{bmatrix} 3 & 2 & 1 \\ 0 & -2 & 3 \\ 0 & -5 & 5 \end{bmatrix}$$</center>

<center>$$\begin{bmatrix} 1 & 0 & 0 \\ -\frac{5}{2} & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} 3 & 2 & 1 \\ 0 & -2 & 3 \\ 0 & -5 & -5 \end{bmatrix} = \begin{bmatrix} 3 & 2 & 1 \\ 0 & -2 & 3 \\ 0 & 0 & -\frac{5}{2} \end{bmatrix}$$</center>

최종적으로 우항에 GE가 완성된 형태로 만들었다. 정리해보면, 첫 번째 행렬 $$\begin{bmatrix} 3 & 2 & 1 \\ 6 & 2 & 5 \\ 9 & 1 & 8 \end{bmatrix}$$ 에 총 3개의 행렬을 순차적으로 곱함으로써, GE 결과물에 해당하는 행렬 $$\begin{bmatrix} 3 & 2 & 1 \\ 0 & -2 & 3 \\ 0 & 0 & -\frac{5}{2} \end{bmatrix}$$ 를 얻었다.  원래 행렬 $A$의 좌측에 총 3개의 행렬 $$X, Y, Z$$를 곱함으로써 GE 결과 행렬 $$A^{\prime}$$ 을 얻었다.
<center>$$\begin{bmatrix} 1 & 0 & 0 \\ -\frac{5}{2} & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -3 & 0 & 1 \end{bmatrix}\begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} 3 & 2 & 1 \\ 6 & 2 & 5 \\ 9 & 1 & 8 \end{bmatrix} = \begin{bmatrix} 3 & 2 & 1 \\ 0 & -2 & 3 \\ 0 & 0 & -\frac{5}{2} \end{bmatrix}$$</center>
<center>$$(ZYX)A = A^{\prime}$$</center>

위의 세 개 행렬을 곱한 $L$행렬($$(ZYX)^{-1}$$) 을 양변의 좌측에 곱해주면, 마찬가지로 $$ A = LU $$로 표현됨을 알 수 있다. 결과적으로, 사이즈가 더 큰 행렬들에 대해서도 lower triangular matrix를 연속적으로 계속 곱함으로써, upper triangular matrix를 만들어낼 수 있음을 알 수 있다.

## LDU Decomposition
$\ $ LDU 분해는, 위에서 진행한 LU 분해에서 한 발자국 더 나아간 것인데, LU 분해에서 얻은 upper triangular matrix에 해당하는 matrix U 를 diagonal matrix인 D와, 새로운 upper triangular matrix U로 분해하는 것이다. U 에 해당하는 $$\begin{bmatrix} 3 & 4 \\ 0 & -3 \end{bmatrix}$$을 분해해보면 아래와 같다.
<center>$$\begin{bmatrix} 3 & 4 \\ 0 & -3 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 0 & -3 \end{bmatrix}\begin{bmatrix} 1 & \frac{4}{3} \\ 0 & 1 \end{bmatrix}$$</center>
결과적으로 D 행렬과 U 행렬로 분해됐으며, 위와 같은 분해가 항상 가능하다는 것은 직접 손으로 분해해보면 쉽게 깨닫는다.
위에서 진행했던 LU decomposition 과정을 LDU 로 확장시키며 행렬 분해에 대한 내용을 끝마친다.
<center>$$\begin{bmatrix} 3 & 4 \\ 6 & 5 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} 3 & 4 \\ 0 & -3 \end{bmatrix}$$</center>

<center> $$ X = LU $$ </center>

<center>$$\rightarrow \begin{bmatrix} 3 & 4 \\ 6 & 5 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} 3 & 0 \\ 0 & -3 \end{bmatrix}\begin{bmatrix} 1 & \frac{4}{3} \\ 0 & 1 \end{bmatrix}$$</center>

<center> $$ X = LDU $$ </center>

> references
https://math.stackexchange.com/questions/266355/necessity-advantage-of-lu-decomposition-over-gaussian-elimination
