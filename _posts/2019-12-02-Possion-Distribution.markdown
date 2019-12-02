---
title: " [확률기초] Poisson Distribution"
tags: Probability Distribution Poisson Binomial
---
<center>$$\lambda = np, n : total trials, p : times the chance of success for each of those trials$$</center><br>
> ex) $$ n = 50, p = 2/5, \lambda = 20 $$

* Binomial Distribution
<center>$$B(n,p) = p(X = k) = {n \choose k}p^k(1-p)^{n-k}$$</center>
여기에서, $$\lambda = np$$ 였으므로, $$p = \lambda/n$$을 대입 후, $$ n \rightarrow \infty$$ 취해주면 다음과 같다.<br>
<center>$$\lim_{n \to \infty}P(X = k) = \lim_{n \to \infty}\frac{n!}{k!(n-k)!}(\frac{\lambda}{n})^k(1-\frac{\lambda}{n})^{(n-k)}$$</center>
<center>$$= \frac{\lambda^k}{k!}\lim_{n \to \infty}\frac{n!}{(n-k)!}\frac{1}{n^k}(1-\frac{\lambda}{n})^n(1-\frac{\lambda}{n})^{-k}$$</center>
