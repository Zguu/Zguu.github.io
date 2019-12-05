---
title: " [확률기초] Poisson Distribution"
tags: Probability Distribution Poisson Binomial
---

# 푸아송 분포의 확률질량함수
$\ $ 기초적인 확률분포 중 하나인 푸아송 분포의 확률질량함수를 유도해보자. 푸아송분포의 정의만 잘 숙지하면 생각보다 간단하게 이항분포 확률질량함수로부터 식을 유도할 수 있다.<br>
푸아송분포는, 특정 시간 $t$내에 사건이 몇 번 일어날 지에 대한 확률분포이다. 이항분포에서 시행 수 $n$이 무한하게 크고, 각 시행에서 사건이 일어날 확률 $p$가 충분히 작다면, 기댓값 $\lambda$는 단순히 $n$과 $p$의 곱일 것이다.

<center>$$\lambda = np$$</center>
<center>$$n = total\ trials, p = times\  the\  chance\  of\  success\  for\  each\  of\  those\  trials$$</center><br>
> ex) $$ n = 50, p = 2/5, \lambda = 20 $$

* Binomial Distribution
<center>$$B(n,p) = p(X = k) = {n \choose k}p^k(1-p)^{n-k}$$</center>
여기에서, $$\lambda = np$$ 였으므로, $$p = \lambda/n$$을 대입 후, $$ n \rightarrow \infty$$ 취해주면 다음과 같다.<br>
<center>$$\lim_{n \to \infty}P(X = k) = \lim_{n \to \infty}\frac{n!}{k!(n-k)!}(\frac{\lambda}{n})^k(1-\frac{\lambda}{n})^{(n-k)}$$</center>
<center>$$= \frac{\lambda^k}{k!}\lim_{n \to \infty}\frac{n!}{(n-k)!}\frac{1}{n^k}(1-\frac{\lambda}{n})^n(1-\frac{\lambda}{n})^{(-k)}$$</center>
위의 마지막 줄의 식을 세 부분으로 나누어서 극한 계산을 진행해보자.
<center>$$\lim_{n \to \infty}\frac{n!}{(n-k)!} \cdots (1)$$</center>
<center>$$\lim_{n \to \infty}(1-\frac{\lambda}{n})^n \cdots (2)$$</center>
<center>$$\lim_{n \to \infty}(1-\frac{\lambda}{n})^{-k} \cdots (3)$$</center>
여기에서, (1)번 식은 다음과 같이 정리된다.
<center>$$\lim_{n \to \infty}\frac{n(n-1)\cdots (n-k+1)}{n^k} = 1$$</center>
해당 식은 분모와 분자의 $n$의 차수가 같으므로, 간단히 1로 수렴한다.<br>
(2)번에 해당하는 식은 다음과 같이 변형할 수 있다.
<center>$$\lim_{n \to \infty}\left\{(1-\frac{\lambda}{n})^{(-\frac{n}{\lambda})}\right\}^{(-n)}$$</center>
<center>$$ = e^{-\lambda}$$</center>
마지막으로, (3)번에 해당하는 식은 $$ -\frac{\lambda}{n} $$ 값이 0으로 수렴하므로 결과적으로 1로 수렴한다.<br>
결과적으로, $$P(\lambda, k) = \frac{\lambda^ke^{-\lambda}}{k!}$$이 성립한다.
