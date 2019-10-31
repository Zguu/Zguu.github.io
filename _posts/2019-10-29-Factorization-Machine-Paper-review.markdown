---
layout: post
title:  " [논문 리뷰] Factorization Machine"
date:   2019-10-18 21:19:41 +0900
categories: Paper Review
comments: true
share: true
featured: true
---

# Factorization Machines
Steffen Rendle (2010)
***
FM(factorization machine)은 기존의 가장 인기있는 머신 러닝 예측 모델인 SVM을 대체할 수 있다고 한다. SVM이 일반적으로 예측 능력이 떨어지는 것은, sparse data 하의 복잡한 kernel 공간에서 신뢰도 있는 파라미터들을 학습하지 못하기 때문이다.
* sparse data : Matrix 형태의 데이터 셋이 촘촘하게 모든 데이터가 기록되지 않은 상태(추가 설명 및 일러스트 추가)
* dense data : sparse data의 개념과는 반대로 데이터 셋이 matrix 형태에 빈틈 없이 잘 기록된 경우<br>

SVM가 large sparse data 상에서 취약한 모습을 보이는 것을 보완할 수 있는 것이 Factorization Machine의 큰 장점이라고 할 수 있다. FM은 biased MF, SVD++, PITF, FPMC 등의 기존 CF(collaborative filtering) 에 대한 성공적 접근법들을 포함하는 개념이다.

* FM의 장점 정리<br>
1) SVM이 해내지 못하는 sparse data에서 파라미터 추정 가능<br>
2) 선형적 복잡도를 갖고 있으므로, 매우 큰 데이터 셋에도 적용 가능 (100m Netflix 데이터와 같은)<br>
3) 매우 제한적인 input data에서 작동하는 다른 최신 기술들과 달리, input data 모양에 크게 영향을 받지 않음

---

### Nested variables interaction vs Cross variables interaion<br>



<a href="https://i.imgur.com/GCBQGkr.png"><img src="https://i.imgur.com/GCBQGkr.png" title="source: imgur.com" width="500px" /></a>

---
### 데이터 예시

 Sparse 매트릭스 데이터 예시를 살펴보자. 각각의 행(row)들은 ${x}$
<a href="https://imgur.com/JFxBz4i.png"><img src="https://imgur.com/JFxBz4i.png" title="source: imgur.com" width="500px" /></a>
