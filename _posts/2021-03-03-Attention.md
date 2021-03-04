---
title: " [Deep Learning] Attention "
tags: DeepLearning NLP Timeseries
---

# Attention

Attention 방법론의 논문과 코드를 살펴보던 와중에, animation으로 너무 잘 설명된 포스팅이 있어서 한글로 번역합니다. 하지만 해당 포스팅에는 논문 수식들이 모두 빠져있는 점이 아쉬운데, 논문 원문에 포함 돼 있던 수식을 포함시켜 내용을 조금 더 풍성하게 만들었습니다.

Attention 방법론은 RNN, LSTM 또는 seq2seq 이후에 나온 논문으로, time series, sequential data 들에서 더욱 좋은 성능을 보여줍니다. 대부분의 RNN 계열 알고리즘들이 과거 데이터들을 잘 반영하지 못한다는 취약점을 안고 있는데, 이러한 점을 보완하기 위해 연구 됐습니다.

아래와 같이 매우 긴 문장을 번역하는 작업에 있어서, decoder로 하여금 encoder의 초반 부분을 모두 기억하면서 해석을 하라는 것은 너무 가혹한 일입니다. 그에 따라 당연히 문장이 길수록 성능이 저하됩니다.

![FIgure1](https://i.imgur.com/GdyVRaJ.png)

## 1. Attention : Overview

실제로 우리가 번역을 진행하는 과정을 예를 들어, `seq2seq` 모델과 `seq2seq2 + attention` 을 비교해보겠습니다.

> Intuition : seq2seq<br>
> 우리가 영어로 된 문장을 시작부터 끝까지 쭉 읽었다고 해보자. 우리가 이것을 다 읽고나서 각 단어를 하나 하나 한국어로 옮겨 적는다. 이 경우에, 문장이 너무 길다면 우리는 문장 초반에 읽었던 내용을 잃어버렸을 수도 있다.

위의 `seq2seq` 모델의 치명적인 약점은, 초기에 입력된 데이터들의 경우는 해석 (decoding) 하는 과정에서 간과될 가능성이 높다는 것입니다. 아래의 `seq2seq + attention` 예시를 보겠습니다..

> Intuition : seq2seq + attention<br>
> 마찬가지로 영어로 된 문장을 읽는 데, 이번에는 키워드를 적어가면서 읽고 있다. 우리가 다 읽은 후 단어를 하나 하나 한국어로 번역하는 과정에서, 우리가 적어놓은 키워드를 참고하면서 번역을 진행한다.

`attention` 방법론은 다른 단어들에 각각 다른 focus를 부여하고 이를 score로 할당합니다. 이 score들은 softmaxed 되며, 이 weighted score를 encoder 의 hidden state 에 aggregate한 후에, 해당 hidden state 값들을 모두 더합니다. 이 값이 `context vector` 가 됩니다.

attention layer의 구현은 아래와 같은 4단계로 나눌 수 있습니다.

### Step 0: Prepare hidden states

아래의 figure에서 보여지는 초록색 원에 해당하는 값들은 encoding 과정에서 나오는 hidden states들입니다. 빨간색 원에 해당하는 값은 첫번째 decoder에서 나온 hidden state에 해당합니다. 아래의 그림에서는 4개의 encoder 가 있으며, 마지막 encoder에서 나온 hidden state들을 decoder 가 처리한 값이 빨간색 원들에 해당합니다.

![Figure1.0](https://i.imgur.com/Rn21qwk.gif)

초록색 원에 해당하는 hidden states들은 encoding 과정에서 얻어집니다. 원 논문의 저자는 encoder에 bidirectional RNN (BiRNN)을 사용한다고 설명하고 있습니다. 즉, forward RNN $\overset{\rightarrow}{f}$ 이 단어들을 $x_1$ 부터 $x_{T_x}$ 까지 읽고 계산하며 얻은 hidden states 값들은 $\overset{\rightarrow}{h}_1 , \dots , \overset{\rightarrow}{h}_{T_x}$ 이 됩니다.

마찬가지로 backward RNN $\overset{\leftarrow}{f}$ 가 $x_{T_x}$ 부터 $x_1$까지 거꾸로 읽은 결과는 backward hidden states $\overset{\leftarrow}{h}_1 , ..., \overset{\leftarrow}{h}_{T_x}$ 이 됩니다.

최종적으로 얻는 annotation vector 는 forward hidden states와 backward hidden states를 concatenating 한 $h_j = [\overset{\rightarrow}{h}_j^T ; \overset{\leftarrow}{h}_j^T]^T$ 로 표현됩니다.

### Step 1: Obtain a score for every encoder hidden state

첫번째 decoder 에서 얻은 hidden states (빨간색 원들) 은 벡터 형태로 표현이 돼 있을 것입니다. 이 벡터와 기존의 encoder에서 얻은 hidden states 벡터들을 내적해서, score를 각각 구합니다.

![FIgure1.1](https://i.imgur.com/oJRt9fh.gif)

> 이를 식으로 표현하면 다음과 같다. <br>
> $e_{ij} = a(s_{i-1}, h_j)$

### Step 2: Run all the scores through a softmax layer

위의 Step 1 에서 얻은 score 들을 모두 softmax 함수를 통해 softmaxed score로 변환합니다.

![FIgure1.2](https://i.imgur.com/0lqeh4T.gif)

> 이를 식으로 표현하면 다음과 같다.<br>
> $\alpha_{ij} = \frac{\text{exp}(e_{ij})} {\sum_{k=1}^{T_x} \text{exp}(e_{ik})}$

### Step 3: Multiply each encoder hidden states by its softmaxed score

softmaxed score를 encoder에서 얻은 hidden state 벡터에 곱해줍니다. 이 벡터를 `alignment vector` 또는 `annotation vector` 라고 칭합니다.

![FIgure1.3](https://i.imgur.com/EINO31S.gif)

> pape $c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j$

### Step 4: Sum up the alighnment vectors

위에서 얻은 모든 alignment vector들을 단순히 합해서, `context vector` 를 생성합니다.

![FIgure1.4](https://i.imgur.com/NEV0Clo.gif)

### Step 5: Feed the context vector into the decoder

위에서 얻은 `context vector` 를 decoder 로 연결시킵니다.

![FIgure1.5](https://i.imgur.com/nWkOZ4q.gif)

Step 1 부터 5 까지를 하나로 합치면 아래와 같습니다.
![FIgure1.6](https://i.imgur.com/stA5DGN.gif)

원문에 있는 Figure와 비교해서 이해해보면 좋습니다.
![스크린샷 2021-03-03 오후 8.54.23](https://i.imgur.com/oyBu3xh.png)


## references <br>
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attn: Illustrated Attention
](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
