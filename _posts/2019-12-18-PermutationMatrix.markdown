---
title: " [선형대수학] Permutation Matrix"
tags: LinearAlgebra Permutation Matrix
---

# Permutation Matrix
$\ $수학 특히, 행렬 이론에서 ***Permutation matrix*** 는 모든 행과 열에 1이 한 번씩만 기록되는 정사각형 형태의 binary matrix (1이 아닌 원소는 모두 0) 중 하나이다. $P$ 로 표현되는 이 행렬은 다른 행렬 $A$ 와 곱해졌을 때($$PA$$ 또는 $$PA$$ ) 해당 행렬 $$A$$ 의 행 또는 열의 원소들의 순서를 바꾸는 역할을 한다.
## Definition
$\ $ 다음과 같이 $m$ 개의 원소들이 정의역으로 주어지면 $m$ 개의 치역을 반환하는 단순한 함수의 형태이다.
<center>$$\pi : \left\{ 1,...,m \right\} \rightarrow \left\{1,...,m\right\}$$</center>
아래와 같이 표현될 수도 있다.
<center>$$\begin{pmatrix} 1 & 2 & \cdots & m \\ \pi(1) & \pi(2) & \cdots \pi(m) \end{pmatrix}$$</center>
$\ $ $m x m$ 형태의 Permutation matrix $$P_\pi = (p_{ij})$$ 는 다음과 같이 표현된다.
<center>$$ P_\pi = \begin{bmatrix} \mathbf{e}_{\pi(1)} \\\mathbf{e}_{\pi(2)} \\\mathbf{e}_{\pi(3)} \\\mathbf{e}_{\pi(4)} \\\mathbf{e}_{\pi(5)} \end{bmatrix} = \begin{bmatrix} \mathbf{e}_1 \\ \mathbf{e}_2 \\ \mathbf{e}_3 \\ \mathbf{e}_4 \\ \mathbf{e}_5 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 & 1 \end{bmatrix}. $$</center>
