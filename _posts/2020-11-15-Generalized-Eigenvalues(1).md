---
title: " [선형대수학] Generalized Eigenvector (1)"
tags: LinearAlgebra
---

# Generalized Eigenvalues and Eigenvector

말그대로 조금 더 일반적인 형태의 eigenvalues, eigenvectors 에 대해서 이해해보자. 해당 형태는 Reyleigh Ratio 값과 비슷한 형태를 보여주므로, Reyleigh Ratio에 대한 리뷰를 먼저 시작해보겠습니다.

## Reyleigh Ratio
Reyleigh Ratio는 다음과 같은 형태를 보여줍니다. SVD에서 first, second laregest singular value, 그에 대한 기하학적 해석에서 한 번 다룬적이 있습니다.

$$ R(x) = \frac{x^TSx}{x^Tx}$$

left singular vector, singular values, right singular vector를 구하는 과정에서, $A^TA$ 또는 $AA^T$ 형태로 곱해진 symmetric 행렬 $S$의 eigenvalue, eigenvector를 구했었는데요. 위의 식에서 보이는 $S$가 이런 symmetric 행렬 중 하나입니다. 또한 이 $R(x)$를 최대화 시키는 벡터는, 첫번째 singular vector 에 해당합니다.

즉, $R(x)$에  $q_1$ 을 대입하면, $R(x)$의 최댓값을 구할 수 있습니다. 지금까지 한 이야기를 수식으로 표현하면 다음과 같습니다.

$$\text{maximum} \quad R(x) = R(q_1) \\ = \frac{q_1^TSq_1}{q_1^Tq_1} = \frac{q_1^T\lambda_1q_1}{q_1^Tq_1} =
\lambda_1$$

이 이야기를 조금 더해보면, $R(x)$를 최소화 시키는 벡터는 마지막 singular vector $q_n$일 것입니다.
> singular vector가 총 n개 존재하는 상황일 때

또한, 모든 $q_k$에 대하여, $R$을 편미분하면 모두 0에 해당하는 값을 얻습니다. 이는 모든 $q_k$에서 함수 $R$은 saddle point에 해당한다는 것으로 이해할 수 있습니다.

symmetric 행렬 $S$를 얻기 전 원래 함수인 $A$의 norm은 다음과 같이 Reyleigh ratio와 관련지어 이해할 수 있습니다.

$$ ||A||^2 = \text{max} \frac{||Ax||^2}{||x||^2} = \text{max} \frac{x^TA^TAx}{x^Tx}\\ = \text{max} \frac{x^TSx}{x^Tx} = \lambda_1(S) = \sigma_1^2(A)$$

## Generalized Eigenvalues and eigenvectors
위의 Reyleigh ratio 수식을 약간 변경하여 Generalized Eigenvalues and eigenvectors 를 표현해보겠습니다. 이러한 형태로 표현하는 것은 모든 수학, 공학 분야에서 일반적으로 쓰입니다. Mass Marix or Intertia Matrix or Convariance Matrix로 불려지는 $M$ 행렬을 분모에 포함시킵니다.

$$ R = \frac{x^TSx}{x^TMx}$$

이와 같이 표현했을 때, 일반적인 Reyleigh Ratio 형태의 식을 eigenvector 형태로 표현한 아래의 식은

$$ Sx = \lambda x$$

다음과 같이 표현됩니다.

$$ Sx = \lambda Mx$$

만약 행렬 $M$이 positive-definite 조건을 만족시킨다면, Max$(R(x)) = \text{eigenvalue of}\space M^{-1}S$가 됩니다.

> 기존 $$\text{maximum} \quad R(x) = R(q_1) = \frac{q_1^TSq_1}{q_1^Tq_1} = \frac{q_1^T\lambda_1q_1}{q_1^Tq_1} =
\lambda_1$$ 와 비교해보면 좋습니다.

위와 같이 표현한 $Sx = \lambda Mx \dots (1)$ 를 조금 더 간단한 형태로 다음과 같이 표현해보려고 합니다.

$$Hy = \lambda y$$

(1)에 해당하는 식의 양변 좌측에 $M^{-1}$을 곱하게 되면, $H$를 $M^{-1}S$로 치환하는 것이 적절해 보이는데요, 하지만 이렇게 $H = M^{-1}S$ 로 치환하는 것은 적절하지 않습니다. 이것은 $M^{-1}S$가 symmetric matrix가 아니기 때문인데, 아래에서 확인할 수 있습니다.

$$M^{-1}S = \begin{vmatrix}m_1 & 0 \\ 0 & m_2 \end{vmatrix}^{-1}\begin{vmatrix} a&b\\c&d  \end{vmatrix} \\ = \begin{vmatrix} \frac{a}{m_1} & \frac{b}{m_1} \\ \frac{b}{m_2} & \frac{c}{m_2}\end{vmatrix} \quad \text{not symmetric}$$

그래서 대안으로 $H$를 표현하는 방법은 다음과 같습니다.

$$H = M^{-\frac{1}{2}}SM^{-{\frac{1}{2}}} \\ =  \begin{vmatrix} \frac{a}{m_1} & \frac{b}{\sqrt{m_1m_2}} \\ \frac{b}{\sqrt{m_1m_2}} & \frac{c}{m_2}  \end{vmatrix} \quad \text{symmetric}$$

여기에서, $M$은 원래 정의부터 symmetric 행렬이었기 때문에, $Q\Lambda Q^T$ 로 표현할 수 있습니다. 만약 $\Lambda > 0$ 이 성립한다면, $M^{\frac{1}{2}} = Q\Lambda^{\frac{1}{2}}Q^T$ 에서 $\Lambda^{\frac{1}{2}} > 0 $이 성립합니다.

정리 : $H = M^{-\frac{1}{2}}SM^{-{\frac{1}{2}}}$ 로 표현하자.

이를 이용해서 다음의 eigenvalue problem 을 풀어보겠습니다.

problem 1) solve $Sx = \lambda MX \quad \\ \text{when} \space S = \begin{vmatrix} 4 & -2 \\ -2 & 4 \end{vmatrix} , M = \begin{vmatrix} 1 & 0 \\ 0 & 2 \end{vmatrix}$

sol 1) $(S-\lambda M)X_1 = 0\\ det(S-\lambda M) = 0 \\ det(\begin{vmatrix} 4-\lambda & -2 \\ -2 & 4-2\lambda \end{vmatrix}) = 0 \\ (4-\lambda)(4-2\lambda) - 4 = 0 \\ \rightarrow \lambda  = 3 \pm \sqrt{3}$

sol 2) $H = M^{-\frac{1}{2}}SM^{-{\frac{1}{2}}} \\ = \begin{vmatrix} 1 & 0 \\ 0 & \frac{1}{\sqrt{2}}\end{vmatrix} \begin{vmatrix} 4 & -2 \\ -2 & 4 \end{vmatrix} \begin{vmatrix} 1 & 0 \\ 0 & \frac{1}{\sqrt{2}} \end{vmatrix} \\ = \begin{vmatrix} 4 & -\sqrt{2} \\ -\sqrt{2} & 2 \end{vmatrix} \\ det(H-\lambda I) = 0 \\ \text{also}, \lambda  = 3 \pm \sqrt{3}$

위에서 볼 수 있듯이 어떤 방법으로 이 eigenvalue problem을 접근해도 같은 결과를 얻는 것을 알 수 있습니다.
