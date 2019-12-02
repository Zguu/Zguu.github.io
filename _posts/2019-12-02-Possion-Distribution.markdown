---
title: " [확률기초] Poisson Distribution"
tags: Probability Distribution Poisson Binomial
---
<center>$$\lambda = np, n = total\ trials, p = times\ the\ chance\ of\ success\ for\ each\ of\ those\ trials$$</center><br>
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
해당 식은 분모와 분자의 $n$의 차수가 같으므로, 간단히 1로 수렴한다.
(2)번에 해당하는 식은 다음과 같이 변형할 수 있다.
<center>$$\lim_{n \to \infty}\left\{(1-\frac{\lambda}{n})^{(-\frac{n}{\lambda})}\right\}^(-n)$$</center>
