---
title: " [선형대수학] Permutation Matrix"
tags: LinearAlgebra
---

# Permutation Matrix
$\ $수학 특히, 행렬 이론에서 ***Permutation matrix*** 는 모든 행과 열에 1이 한 번씩만 기록되는 정사각형 형태의 binary matrix (1이 아닌 원소는 모두 0) 중 하나이다. $P$ 로 표현되는 이 행렬은 다른 행렬 $A$ 와 곱해졌을 때($$PA$$ 또는 $$PA$$ ) 해당 행렬 $$A$$ 의 행 또는 열의 원소들의 순서를 바꾸는 역할을 한다.
## Definition
$\ $ 다음과 같이 $m$ 개의 원소들이 정의역으로 주어지면 $m$ 개의 치역을 반환하는 단순한 함수의 형태이다.
<center>$$\pi : \left\{ 1,...,m \right\} \rightarrow \left\{1,...,m\right\}$$</center>
아래와 같이 표현될 수도 있다.
<center>$$\begin{pmatrix} 1 & 2 & \cdots & m \\ \pi(1) & \pi(2) & \cdots & \pi(m) \end{pmatrix}$$</center>
$\ $ $m\ by\ m$ 형태의 Permutation matrix $$P_\pi = (p_{ij})$$ 는 다음과 같이 표현된다.<br>
예를 들어 아래의 permutation matrix $$P_\pi$$는 다음과 같은 permutation 에 해당한다 : $$ \pi = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 1 & 4 & 2 & 5 & 3\end{pmatrix}$$
<center>$$ P_\pi = \begin{bmatrix} \mathbf{e}_{\pi(1)} \\\mathbf{e}_{\pi(2)} \\\mathbf{e}_{\pi(3)} \\\mathbf{e}_{\pi(4)} \\\mathbf{e}_{\pi(5)} \end{bmatrix} = \begin{bmatrix} \mathbf{e}_1 \\ \mathbf{e}_2 \\ \mathbf{e}_3 \\ \mathbf{e}_4 \\ \mathbf{e}_5 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 & 1 \end{bmatrix}. $$</center>
위의 가장 오른쪽 binary matrix 형태에서, 열을 기준으로 볼때 1이 몇 번째에 나타는 지를 통해, 원소 나열의 순서가 어떻게 변하는지 확인할 수 있다. <br>
$$P_\pi$$의 첫번째 열을 보면 첫번째 원소가 1이기 때문에, $$ \pi = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 1 & 4 & 2 & 5 & 3\end{pmatrix} $$ 행렬의 첫 행의 첫 원소 1은 변하지 않고 두번째 행의 첫 원소에 위치한다. <br>
하지만, $$P_\pi$$의 두번째 열에서는 1이 세번째 원소에서 나타나고, $$ \pi = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 1 & 4 & 2 & 5 & 3\end{pmatrix} $$ 에서 확인해보면 첫 행의 2라는 원소가 두번째 행에서는 세번째 원소로 위치함을 알 수 있다. 이와 같이 permutation matrix가 순서 변경 역할을 해냄을 이해할 수 있다. <br>
$\ $위의 예시에서, 각 $$\mathbf{e}_j$$는 $m$ 의 길이를 가진 row vector 들이며 ***standard basis vector*** 라고 부른다. 해당 열에서 1이 나타나는 위치가 permutation 변환 이후 원래 값의 위치 변경을 나타내주므로, 위와 같은 $$P_\pi$$를 ***column representation*** 이라 부른다. 물론 비슷한 방법으로, ***row representation*** 도 가능하나, 해당 표기는 건너뛴다. 간단하게 해당 표현 방법을 떠올려보고 넘어가자.

## Example
$\ $ $$P_\pi$$가 $$A$$ 와 곱해지는 과정에서 왼쪽 또는 오른쪽 어디에 위치하느냐에 따라 ***column representation*** 인지, ***row representation*** 인지 결정된다. 일반적으로는 왼편에 곱해져서 $$PA$$의 형태를 띄는 경우, ***column representation***, 오른편에 곱해져서 $$AP$$의 형태를 띄는 경우는 ***row representation*** 에 해당한다. 아래의 그림을 참고하자.
<center><img src="https://imgur.com/NacjmsF.png" width="80%" height="80%"></center>

> reference : https://en.wikipedia.org/wiki/Permutation_matrix
