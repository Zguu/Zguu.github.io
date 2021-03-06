---
title: " [베이지안 추론] Bernoulli Likelihood with hierarchical Prior"
tags: Bayesian
---

# Chapter 9. Bernoulli Likelihood with hierarchical Prior
$\ $이전 챕터에서는, 파라미터들이 서로 독립적이라는 가정하에 샘플링을 진행했지만, 이번 챕터에서는 두개 또는 그 이상의 파라미터들이 서로 의존적 관계일 때를 살펴본다. 예를 들어, 우리는 동전이 주조되는 공장에 따라 동전의 bias가 달라진다고 가정해보자. 우리는 주조 과정이 갖는 파라미터 값에 대한 사전 믿음을 갖고 있고, 주조를 진행하는 파라미터들에도 동전의 bias가 의존한다는 사전 믿음을 갖고있다. 그리고 우리는 동전을 몇번 던진 후에 얼마나 heads가 나오는 지를 관측한다. 데이터들은 우리의 동전의 bias에 대한 믿음에 영향을 끼친다. 이뿐만 아니라, 동전의 bias가 주조 파라미터에 의존할 것이라는 믿음 자체에 대해서도 영향을 끼친다. 또한 주조 파라미터 값들 자체에 대한 우리의 믿음도 영향을 끼친다.
>말이 좀 복잡해지는데.. 다른 예를 들어보자. 우리는 로또를 맞을 확률이 매주 서로 독립이라는 것을 이미 알고 있다. 그런데 어느날부터 로또 당첨 방송이 사실 조작이라는 의심을 품게된다. 즉, 지금까지의 로또 1등 당첨결과가 이번주에도 영향을 끼칠 것이라고 믿지 시작한 것이다 (실제로 그럴 일은 없다). 의심을 품기 시작한 시점에 나는 이 방송이 조작일 것이라고 10%정도로 의심했다. 하지만 매주 데이터들을 보면서 나의 이 의심 자체에도 의구심이 들어서 의심의 정도가 5%로 내려갔다. 그런데, 내가 이 로또 방송을 의심하게 만든 다른 변수(뭐...예를 들어 주변에서 2번 연속 당첨받은 사람?) 값 자체에도 의심이 생기기 시작했다. 예시가 적절한지 모르겠다. (난 잘한 것 같은데)

데이터들 자체에도 직접적으로 영향을 끼치는 파라미터들은 그냥 여전히 파라미터(parameter)라고 부르자. 다만, 우리의 믿음에 영향을 줌으로써 데이터에 영향을 끼치는 파라미터들은 하이퍼 파라미터(hyper parameter)라고 부르자. 사실 어떤 관점에서 보면 일반 파라미터와 하이퍼 파라미터간에는 차이가 없다. <br>
## 9.1 A single coin from a single mint
$\ $동전의 평향도 $\theta$는 head를 얻을 확률을 결정하며 그에 대한 베르누이 분포는 우리가 알듯이 아래와 같다.
!['Img'](https://imgur.com/Q2S7upQ.png)
$\ $<br>
bias에 대한 사전 분포는 $p(\theta)$로 표현 되는데, 현재 예시에서, 이 사전분포가 베타분포라고 가정했다. 베타분포는 a,b 두개의 파라미터를 가지며 다음과 같이 정의됨을 안다: beta$$(\theta,a,b) = \theta^{(a-1)} (1-\theta)^{(b-1)}/B(a,b)$$ 이 베타 분포를 조금 더 직관적으로 만들기 위해, 상응하는 평균 $\mu$, 샘플 수 $K$를 이용해서 표현해보자. $$a = \mu K$$, $$b = (1-\mu)K$$임을 이미 배웠다. 따라서 다음과 같이 표현이 가능하다.
!['Img2'](https://imgur.com/05EFpje.png)
$\ $<br>
!['Img3'](https://imgur.com/tQzFcSK.png)
### 9.1.1 Posterior via grid approximation
$\ $파라미터들과 하이퍼파라미터들이 유한한 범위에 퍼져있고, 차원 수가 너무 높지 않을 때, 사후분포를 grid 근사를 통해 얻을 수 있다. 우리는 현재 유한한 범위 [0, 1] 에서 $\theta$ 와 $\mu$를 갖고 있다. 그러므로, grid 근사가 가능하며 해당 분포는 그래프로 그려질 수 있다. 아래의 Figure 9.2는 하이퍼 파라미터가 수식 (9.3)의 베타 분포 형태를 갖는 케이스를 보여준다. $$A_{\mu} = 2, B_{\mu} = 2$$ 이며, $$p(\mu) = beta(\mu|2,2)$$라고 볼 수 있다.
!['Img4'](https://imgur.com/aeqnLm5.png)
!['Img5'](https://imgur.com/ZEvcNNg.png)
<br>
<br>
$\ $이 하이퍼 사전분포 (hyperprior)는 주조기의 $\mu$값이 .5 주변에 있다는 것을 표현하지만 높은 불확실성을 갖고 있다. 이 베타분포는 Figure 9.2의 최상단 오른쪽에 나타난다. 이 그래프는 수직 축이 $\mu$인 형태에서 측면으로 기울어져 있다; 이 방향은 Figure 전체에서 $\mu$가 같은 방향을 가르키는 것을 가능하게 해주며 panel들간의 비교를 용이하게 할 수 있다.
$\ $ $\mu$에 대한 $\theta$ 값의 사전분포는 또 다른 베타 분포로 표현 된다. $$K = 100$$이며, $$p(\theta|\mu) = beta(\theta, \mu 100,(1-\mu)100)$$ 으로 표현된다. 사전분포는 다른 분포들보다 상대적으로 낮은 불확실성을 보여주며, $\theta$ 편향도가 $\mu$에 가까움을 보여준다. 이 분포들은 Prior Figure의 두번째 행의 오른쪽 패널에 나타난다. 왼쪽과 가운데에 있는 패널들은 $joint$ 사전분포 $$p(\theta,\mu) = p(\theta|\mu)p(\mu)$$를 나타낸다. 가운데에 있는 contour plot은 왼쪽에 있는 3차원 그래프를 위에서 내려다본 것을 보여준다. 이것이 grid 근사이기 때문에, joint 사전분포 $$p(\theta,\mu)$$는 각각의 grid 포인트들에서 $$p(\theta|\mu)$$와 $$p(\mu)$$를 곱함으로써 계산됐다. 이후에 각 grid 점들에서 이산확률질량 값으로 표현하기 위해 전체 grid를 더한 값으로 나눠주어야 한다.
$\ $ normalizing이 완료된 각 확률질량들은 각 grid 점들에서 확률 밀도로 변환되기 위해, 각 확률 질량을 grid cell의 면적으로 나눠준다. 최상단 오른쪽에 있는 plot ($\mu$ 축으로 사이드에 붙어있는)은 사전 분포의 $marginal$ 분포를 보여준다.<br>
$\ $Figure 9.2의 가운데 줄은 파라미터 공간에 걸쳐져 있는 가능도 분포를 보여준다. 데이터 $D$는 9개의 heads, 3개의 tails로 구성돼있다. 가능도 분포는 베르누이 분포들의 곱 $$p(D|\theta) = \theta^9 (1-\theta)^3$$ 이다. 해당 그래프에서 모든 Contour line들이 $\mu$ 축에 평행이며, $\theta$ 축에는 수직임을 알 수 있다.즉, 이 가능도 함수가 $\mu$에는 전혀 영향을 받지 않고 오직 $\theta$에 의한 함수라는 것을 알 수 있다.<br>
$\ $Figure 9.2의 네번째 줄에 보이는 사후분포는 각 grid 점들에서 joint prior와 likelihood 값들을 곱함으로써 얻어진다. point-wise 곱들은 해당 파라미터 공간에 걸친 값들의 전체 합으로 나누어짐으로써 normalizing된다. 특정 $\mu$ 값에서 사후분포를 수직으로 잘라낸다고 생각하면, 조건부 분포인 $$p(\theta|\mu,D)$$를 얻을 수 있다. Figure 9.2에서 볼 수 있는 사전분포 $$p(\theta|\mu)$$의 그래프와 사후분포 $$p(\theta|\mu,D)$$ 그래프 사이에 큰 차이가 없다는 점에 주목하자. 이는, 애초에 $\theta$가 $\mu$에 의존할 것이라는 것 자체에 대한 불확실성이 존재했기 때문이다.(와우111) 네번째 줄의 우측에 있는 분포 $p(\mu|D)$가 어떻게 나온 것인지 지에 대해 깨닫고 넘어가자. 앞에서 사전분포 $$p(\theta|\mu)$$의 그래프와 사후분포 $$p(\theta|\mu,D)$$ 그래프는 서로 큰 차이가 없었던 것에 반해, 사전분포 $$p(\mu)$$와 사후분포 $$p(\mu|D)$$는 꽤나 다름을 보여준다. 우리가 관측한 데이터가, 어떻게 $\mu$가 분포하는 것에 대한 믿음에 영향을 주었다고 볼 수 있다. 이는 사전분포가 매우 불확실했기 때문에, 데이터에 의해 영향을 받아 쉽게 변형된다는 것을 의미한다. (와우222)<br>
$\ $지금까지의 케이스들과 대조적으로 만약에 $\mu$에 관한 사전 믿음에 우리가 더 높은 확실성을 갖고 있다고 생각해보자. 하지만 $\mu$에 대한 $\theta$의 의존성 자체에는 낮은 확실성을 갖고있다. Figure 9.3은 이러한 경우를 잘 보여준다.
!['Img6'](https://imgur.com/kHPYIxb.png)
!['Img7'](https://imgur.com/ZBICPdX.png)
<br>
<br>
$\ $현재 케이스에서, $$p(\mu) = beta(\mu|20,20)$$이고, $$p(\theta|\mu) =  beta(\theta|\mu 6, (1-\mu)6)$$ 으로 볼 수 있다. 이 경우에, 최상단 행 오른쪽에 있는 $$p(\mu)$$는 이전에 비해 적은 불확실성을 보여주지만, $$p(\theta|\mu)$$는 더욱 넓은 모습을 보여준다. Figure 9.2에서와 같은 데이터를 사용했기 때문에, likelihood 그래프는 같은 모습을 보여준다. 사전분포와 사후분포를 비교해보자. $\mu$에 대한 분포는 거의 변하지 않았는데 이는 $\mu$ 자체에 대한 확실성이 높아졌기 때문이다. 하지만, $$p(\theta|\mu,D)$$는 사전분포와 매우 다른데, 이는 이전의 확실성이 줄어들었기 때문에 이러한 분포에 큰 영향을 끼친 것이다. 이 경우에, 데이터는, 우리가 처음에 의심했던 것과는 다르게 $\theta$값이 $\mu$에 의존한다는 것을 보여준다.<br>
$\ $ 요약해보자면, 데이터는 (1) 하이퍼파라미터들에 대한 우리의 믿음에 영향을 끼친다. 그리고 (2) 하이퍼파라미터에 대한 파라미터의 의존성에 대한 우리의 믿음에 영향을 끼친다.
>지금부터 나오는 문장들이 parsing(번역) 자체가 어려울 것이라고 하는데, 저자는 충분히 노력할 만한 개념이라고 집중하라고 한다.(집중)

* 하이퍼 파라미터에 대한 사전분포는 불확실성이 높지만, 파라미터가 하이퍼 파라미터에 결국 영향을 끼칠 것이라는 의존성에 대한 불확실성은 낮은 경우, 데이터는 불확실성이 높은 '하이퍼 파라미터에 대한 사전분포' 의 믿음에 영향을 끼친다. <br>
위의 이야기는 Figure 9.2에서 우리가 살펴본 것이다. <br>
* 하이퍼 파라미터에 대한 사전분포는 확실성이 높지만, 파라미터가 하이퍼 파라미터에 결국 영향을 끼칠 것이라는 의존성에 대한 확실성은 낮은 경우, 데이터는 불확실성이 높은 '파라미터와 하이퍼 파라미터 간의 의존성'의 믿음에 영향을 친다.<br>
$\ $다르게 얘기하면, 어떠한 사전분포에 관점에서 바라보든, 그것이 불확실하다면 해당 분포는 데이터에 의해 영향을 쉽게 받는다!
## 9.2 Multiple coins from a single mint
$\ $이전 섹션들에서 우리는 $single$ 동전을 던지는 시나리오에서 bias $\theta$와 하이퍼파라미터 $\mu$에 대한 추론을 하는 것을 살펴보았다. 우리는 조금 더 확장해서 생각해보고자 한다: 만약 우리가 한 개 이상의 동전들에서 데이터를 수집했다면 어떨까? 만약 각각의 동전이 자기들만의 bias $$\theta_j$$를 지니고 있다면, 우리는 각각의 동전들에 대한 distinct 파라미터들에 대하여 추정을 진행하는 것이다. 지금까지는 모든 동전들이 같은 주조기에서 생산됐다고 가정했다. 즉, 우리가 모든 동전들에 대해 같은 사전 믿음 $\mu$를 갖고 있었다는 것을 의미한다. 또한, 각 동전들이 서로에게 독립적이라고 가정했다.<br>
$\ $동전 케이스들 너무 진부하니, 현실에서 볼 수 있는 다른 예를 좀 들어보자. 실험에서 치료를 하는 것을 가정해보자. 우리는 특정 약에 대해 관리를 진행하고 있다. 여기서 약들은 주조기에 해당한다. 약을 투여받는 피실험자들은 동전에 해당할 것이다. 피실험자들의 약에 대한 반응은 각각의 bias의 역할을 한다. 서로 다른 피실험자들은 약에 의해 다른 반응을 보이겠지만, 이 반응은 약의 전체적인 효과에 의존할 것이다. 우리는 각 피실험자들이 서로 상호작용하지 않도록 실험을 설계했고, 우리는 각각의 biases들이 서로 독립적이라고 가정할 수 있다. 우리는 피실험자들로부터 베르누이 반응을 측정하면서 동전을 던지는 것과 같다. 예를 들어, 이 약이 기억력에 영향을 끼친다고 가정해보자. 우리는 피실험자로 하여금 특정 단어들을 외우도록 하고, 얼마나 많은 단어를 시간이 지난 후에도 기억하는지로 기억력을 테스트 해볼 수 있다. 특정 단어들은 동전을 던지는 행위에 해당하고, 기억하냐 안하냐는 head 인지 tail인지로 볼 수 있다.<br>
$\ $이 시나리오는 Figure 9.4에 요약 돼있다. 이것은 Figure 9.1과 닮아있지만 한가지 미묘한 차이가 있다. single $\theta$ 값으로 존재하는 것 대신에, 각각의 동전들에 대하여 다른 $\theta$값들이 존재하고 있다. $j^{th}$ 동전의 bias를 $$\theta_j$$로 표현한다. 각각의 동전들을 던지는 것은 다른 동전들에서 진행하는 것이기 때문에, 결과는 double-subscripted 됐다. ($$i^{th}$$ flip of the coins $$j^{th}$$ coin is denoted as $$y_{ji}$$). 모델이 $$ J + 1$$ 파라미터들, $$\theta_1, ..., \theta_J, and \mu$$를 포함하고 있으며 이것들은 모두 동시에 estimated 된 것임을 숙지하자.
!['Img8'](https://imgur.com/n8vIkdS.png)
$\ $하이퍼파라미터 $\mu$에 대한 우리의 믿음이 single value로 좁혀질 때, Figure 9.4 에서 dependency 구조에 어떤 일이 일어나는지 확인해보자. 우리가 가정한 특정한 $\mu$ 값에 아주 좁은 spike가 놓여진 $$p(\mu)$$의 분포로 시각화될 수 있다. 그러므로, $\mu$는 사실상 $constant$이며, $$p(\theta_j)$$는 두 개의 상수 ($\mu$, $K$)에 의한 함수로 볼 수 있다. 우리는 $\mu$에 대한 어떠한 추론도 할 것이 없으므로(이미 $\mu$에 대한 값을 안다), 이 경우에 $\mu$에 대한 하이퍼 레벨을 포함시키는 것은 우리의 추론 과정에 별 도움이 되지 않는다. 그러므로, 우리는 더 이상 hierarchical한 모델을 갖지 않고, 대신에 $J$ 셋의 one-level 모델을 갖게 된다.<br>
$\ $ $\theta_j$의 $\mu$에 대한 의존에 대하여 우리의 믿음이 사실상 $deterministic$ 관계가 됐을 때, dependency 구조에 어떤 일이 일어나는지 숙지하자. 예를 들어, $K$ 값이 무한하게 커져서 $\mu$값이 특정한 값을 갖게 되면, 모든 $\theta_j$는 모두 그 특정 값을 갖게 된다. 이 경우에, $\theta_j$가 갖게되는 유일한 불확실성은 $\mu$의 불확실성이다. 이 경우에, $$\theta_j = \mu$$이고, 같은 $\mu$값들이 모든 $J$개의 동전들에 사용되기 때문에, 우리는 마치 모든 동전들이 같은 기저의 bias를 갖는 것으로 생각할 수 있다.
### 9.2.1 Posterior via grid approximation
$\ $조금 더 구체적인 예시로, 우리가 같은 주조기에서 나오느 두개의 동전을 갖고 있다고 가정해보자. 우리가 biases $\theta_1$, $\theta_2$ 및 주조기의 $\mu$값을 추정하고 싶다고 가정해보자. Figure 9.5와 9.6은 두개의 다른 사전분포들에 대한 grid 근사를 보여준다.
!['Img9'](https://imgur.com/qCq0VUi.png)
!['Img10'](https://imgur.com/pKOEwh8.png)
<br>
위의 Figure 9.5에서 $\mu$ 값은 $$\mu = 0.5$$ 주변에서 솟아있다. 이것은 $$beta(\mu|2,2)$$의 형태이며 즉, Figure 9.4에서 보이는 $$ A_\mu = B_\mu = 2 $$의 형태이다. Figure 9.4의 중간에 보이는 공식에서 K = 5의 경우에 따르면, 동전들의 biases는 $\mu$값에 약한 의존도를 보여준다($$p(\theta_j|\mu) = beta(\theta_j|\mu\cdot 5, (1-\mu) \cdot5)$$). 전체적인 사전 분포는 총 세개의 파라미터들에 대한 joint 분포이다: $$\mu, \theta_1, \theta_2$$. 어떠한 점 $$(\mu, \theta_1, \theta_2)$$에서의 사전확률은 다음과 같다. $$p(\mu)p(\theta_1|\mu)p(\theta_2|\mu). <br>
$\ $Figure 9.5의 가운데 줄은 likelihood 함수의 형태를 보여주며 한 동전에서는 15번의 던지기 시도 중 3번의 head, 다른 동전에서는 5번의 시도 중 4번의 head가 나온 경우이다.<br>
$\ $Figure 9.5의 맨 아랫 줄 그림은 사후 분포를 보여주는데, $\theta_1$에 대한 사후분포는 우리가 얻은 데이터 .2 주변에 중심을 잡고 있으며, $\theta_2$에 대한 사후분포 또한 .8 주변에 중심을 잡고 있다. $\theta_1$이 보여주는 contour는 $\theta_2$가 보여주는 contour에 비해 더 밀도가 높은 모습을 보여주는데, 이는 첫번째 동전에서 더 많은 데이터를 얻은 덕에 불확실성이 줄어들었기 때문이다. $\mu$에 대한 사후분포도 상대적으로 넓게 퍼져있음을 보여준다. 이 결과들은 Figure 9.6 과 비교하면 좋다. Figure 9.6은 똑같은 데이터를 사용했으나, 다른 사전분포를 사용한 것이다. Figure 9.6에서 $\mu$에 대한 사전분포는 여전히 gentle한 peak를 보여주지만, $\theta_j$의 $\mu$에 대한 사전 의존도에 대한 부분은 이전에 비해 믿음이 더욱 강하다고 보여진다. 이러한 의존도는 Figure 9.6의 맨 위 두개의 패널에서 시각적으로 보여진다. ($$p(\theta_j, \mu)$$). 해당 contour들은, $\theta_j$가 $\mu$ 값에서 가깝지 않을 때, 즉, 대각선에서 멀어질 수록 $$p(\theta_j, \mu)$$값이 작아짐을 보여준다.
!['Img11'](https://imgur.com/z5Nu8nd.png)
!['Img12'](https://imgur.com/vkqD07E.png)
$\ $위의 Figure 9.5, 9.6 모두에서 보여진 예시들은 단지 50대 점들의 3차원 $$(\mu, \theta_1, \theta_2)$$ 조합, 즉, $$50^3 = 125,000$$ 점들에서 근사를 진행 한 것이기 때문에, 모든 컴퓨터가 무리 없이 진행 할 수 있는 방법이다. 하지만 동전의 갯수가 추가돼서 변수가 4개로, 5개로 늘어나면 근사해야할 점들은 기하급수적으로 늘어난다. 따라서, Grid 근사법은 차원 수가 높은 경우에 적합하다고 볼 수 없다.
### 9.2.2 Posterior via Monte Carlo Sampling
$\ $
