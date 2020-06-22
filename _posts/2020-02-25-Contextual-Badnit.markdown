---
title: " [추천시스템] A Contextual-Bandit Approach to Personalized News Article Recommendation"
tags: Recommender Bandit Contextual Personalization
---

# Contextual Bandit이 뭐야?
$\ $일반적인 K-armed Bandit (MAB) 문제들과는 다르게, Contextual Bandit은 유저와 arm들에 대한 feature 데이터들을 고려하여 reward 최대화를 진행한다?

## 1.introduction

## 2. Formulation & Related Work

## 3. Algorithm
### 3.1 LinUCB with Disjoint Linear Models
$\ $ 특정한 arm $a$에서 얻을 수 있는 expected payoff의 계산 식은 다음과 같이 구한다. $d$차원의 feature를 갖고 있는 벡터 $$\mathrm{x}_{t,a}$$ 와, 우리가 아직 모르는 coefficient 벡터 $$\theta_{a}^{\ast}$$를 사용해서, 아래와 같이 계산한다.<br>
<center>$$E[r_{t,a}|\mathrm{x}_{t,a}] = \mathrm{x}_{t,a}^T \theta_{a}^{\ast}$$</center>
이 모델은 각각의 arm들이 파라미터들을 공유하지 않으므로, $$disjoint$$ 모델이라고 불려진다. $$\mathbf{D}_a$$ 는 $t$번째 시도에서 $$m x d$$ 의 형태를 갖는 design matrix 이다. 이는 총 $m$ 개의 트레이닝 인풋을 갖고 있음을 의미한다. $$b_a \in \mathbb{R}^m$$은 앞의 트레이닝 인풋 값들에 대응하는 response vector이다. 이는 총 $m$ 개의 click/no click 유저 피드백을 포함한다.
> design matrix로 일컫는 $$\mathbf{D}_a$$는 기존의 regression 문제에서 $$X$$에 해당하는 입력 값에 해당한다. response vector에 해당하는 $$b_a$$는 $$y$$ 벡터에 해당한다고 생각하면 될 듯 하다.

training data에 해당하는 $$(\mathbf{D}_a, \mathrm{c}_a)$$를 ridge regression에 적용하면 아래와 같이 coefficients를 얻는다.<br>
<center>$$\hat{\theta_a} = (\mathbf{D}_a^T \mathbf{D}_a + \mathbf{I}_a)^-1 \mathbf{D}_a^T \mathrm{c}_a \cdots (3)$$</center>
> $$\mathbf{D}_a^T \mathbf{D}_a + \mathbf{I}_a$$ 는 정의 상으로, $$\mathbf{A}_a$$에 해당하며, 이 $$\mathbf{A}_a$$는 추후에 공분산으로서 역할을 한다.

> 여기에서 ridge regression을 적용했는데, 왜 $$\lambda\mathbf{I}_a$$ 가 붙지 않고, $$\mathbf{I}_a$$가 붙는 걸까?
