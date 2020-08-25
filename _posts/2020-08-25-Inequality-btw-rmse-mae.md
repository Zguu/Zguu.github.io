---
title: " [머신러닝] Inequality betwen RMSE and MAE"
tags: MachineLearning
---

# Definition of RMSE & MAE
Regression, Ranking evaluator로 사용되는 Root-Mean Squared Error(RMSE)의 정의는 다음과 같다.
$$RMSE = {\frac{1}{n}\sum\left\{ (X-\hat{X})^2 \right\}}^{\frac{1}{2}}$$
$$= \frac{1}{n}{(X-\hat{X})^T(X-\hat{X})}^{\frac{1}{2}}$$
$$= \frac{1}{n}\left\{{{\sum_i \sum_j (x_{ij} - \hat{x}_{ij})^2}}\right\}^{\frac{1}{2}}$$

더욱 간단한 형태로, Mean Absolute Error(MAE)의 정의는 다음과 같다.
$$MAE = \frac{1}{n}\sum|X-\hat{X}|$$
$$=\frac{1}{n}\sum_i\sum_{j}|(x_{ij} - \hat{x}_{ij})|$$

RMSE, MAE 모두 우리가 예측한 predicdted value가 actual value에서 얼마나 차이가 나는지(error)를 측정하며, 음수 값의 error를 갖는 경우 양수로 변환을 해주는 특징이 있다. 하지만, 이 양수 변환을 RMSE는 제곱을 통해 해결하며, MAE는 절댓값으로 해결한다는 차이가 있는데, 이러한 계산적 특징때문에, error 값이 크면 클수록 RMSE는 민감하게 기하급수적으로 커진다는 특징을 보일 수 밖에 없다.. <br>
그에 반해 abnormaly 하게 큰 error에 대해서 MAE는 상대적으로 덜 민감하게 선형적으로 증가하는 모습을 보일 것이다. 이러한 점들을 이해한다면, RMSE가 그럼 항상 MAE보다 크지 않을까? 라는 궁금증이 생길 수 있다. 하지만, 제곱을 하는 과정에서, RMSE는 제곱을 취하므로, error가 모두 -1과 1사이인 경우에는, 오히려 RMSE가 MAE보다 작지 않을까? 하는 의문이 생길 수 있다. 이번 포스팅을 통해 이러한 RMSE와 MAE 간의 대소관계를 증명해보고자 한다.

## $$MSE\ 와 (MAE)^2 간의\ 대소관계$$
RMSE와 MAE의 관계를 보기전에 MSE와 MAE 제곱 간의 대소관계를 증명해보자. 그 이후에 RMSE와 MAE 간의 대소관계 증명도 자연스럽게 해결될 것이다.
MSE는 미세하게 RMSE와 다르다. 아래의 MSE의 식을 위의 RMSE 식과 비교하여 혼동하기 않도록 하자.
$$MSE = {\frac{1}{n}\sum\left\{ (X-\hat{X})^2 \right\}}$$
$$= \frac{1}{n}{(X-\hat{X})^T(X-\hat{X})}$$
$$= \frac{1}{n}{{\sum_i \sum_j (x_{ij} - \hat{x}_{ij})^2}}$$
> MSE는 RMSE의 제곱이 아니다. MSE는 RMSE의 제곱에 $n$을 곱한 값과 같다.

편의를 위해, $x_{ij} - \hat{x}_{ij}$ 를 $e\ (error)$로 표현하자. 정리하면 다음과 같다.

$$RMSE = \frac{1}{n}\left\{ \sum_{i=1}^n e_{i}^2 \right\}^{\frac{1}{2}}$$
$$MSE = \frac{1}{n}\sum_{i=1}^n e_{i}^2$$
$$MAE = \frac{1}{n}\sum_{i=1}^n e_i$$

이제, 다음을 증명해보자. $MSE \geq (MAE)^2$
즉, 아래의 식 (1)을 증명하는 것이 우리의 목표이다. $$\frac{1}{n}\sum_{i=1}^n e_{i}^2 \geq (\frac{1}{n}\sum_{i=1}^n e_i)^2\  \cdots\   (1)$$
다음과 같이 에러의 평균 $\bar{e}$을 정의해두면, 식을 조금 더 간단히 쓸 수 있다.
$$\bar{e} = \frac{1}{n}\sum_{i=1}^{n} e_i$$
에러 텀의 제곱 합에 해당하는 부분에서 증명을 시작하자.
$$\sum_{i=1}^{n} e_{i}^2 \ = \ \sum_{i=1}^{n}(e_i- \bar{e} + \bar{e})^2$$
$$ = \sum_{i=1}^{n}((e_i - \bar{e})^2 + 2\bar{e}(e_i - \bar{e}) + \bar{e}^2)$$
$$ = \sum_{i=1}^{n}(e_i - \bar{e})^2 + 2\bar{e}(\sum_{i=1}^{n}e_i - n\bar{e}) + n\bar{e}^2$$
$$ = \sum_{i=1}^{n}(e_i - \bar{e})^2 + n\bar{e}^2$$
결과적으로 아래과 같은 식을 얻었으며, 이는 $\sum_{i=1}^{n}e_{i}^{2}$ 와 ${n\bar{e}^2}$ 간의 부등식을 이끌어낸다.
$$\sum_{i=1}^{n}e_{i}^{2} = \sum_{i=1}^{n}(e_i-\bar{e})^2 + n\bar{e}^2 \geq n\bar{e}^2$$
$$\longrightarrow \  \frac{1}{n}\sum_{i=1}^{n}e_{i}^{2} \geq \bar{e}^2 \  \cdots \  (2)$$

$$\therefore MSE \geq (MAE)^2$$
이로써 MSE는 항상 MAE 제곱 이상의 값을 갖는다는 것을 알 수 있다. 위에서 얻은 식 (2)에 양변에 제곱근을 취하면 아래와 같은 식을 얻는다.
$$\sqrt{\frac{1}{n}}\sqrt{\sum_{i=1}^{n}e_{i}^{2}}\geq\bar{e}$$
양변을 $\frac{1}{n}$으로 나누면 아래와 같은 식을 얻는다.
$$\longrightarrow \sqrt{\frac{1}{n}}RMSE \geq MAE\  \cdots \  (3)$$
우리가 (2) 식을 (3)식으로 변형하는 과정에서, 취한 제곱근과 n으로 나누는 과정은 모두 부등호의 방향을 변경시키지 않는다. (데이터의 관측 수를 의미하는 n은 항상 양수이므로)
이로써, RMSE는 일반적인 상황에서 항상 MAE 이상의 값을 갖는다는 증명을 끝낸다.
