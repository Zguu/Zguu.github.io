---
title: " [논문 리뷰] Deep Neural Networks for YouTube Recommendations"
tags: Recommendation NeuralNetwork DNN YouTube
---

# 유튜브 추천 시스템 논문
RecSys '16 September 논문이며 당시 학계에서 많은 주목을 받은 논문이다. 현재 유튜브의 추천 시스템은 이 시스템에서 훨씬 더 발전된 형태일 것이고 그에 대한 공부가 필요할 것이다. 하지만, 기존의 단순한 CF나 MF를 활용한 추천 시스템에서 더욱 나아가, DNN을 활용한 추천 시스템 연구들에 대한 공부를 시작하기에 가장 의미있을 것 같다는 생각을 했다.

---
## Introduction
 유튜브에서의 영상 추천 시스템은 크게 다음과 같은 세가지 난관이 있다.
 - Scale(규모의 문제) : 작은 데이터에서 잘 작용하던 기존의 추천시스템들이 유튜브와 같은 거대한 데이터에서 잘 작용하지 않는 경우가 많다. 유저 데이터와 영상 corpus 데이터가 다른 서비스와는 비교가 되지 않을 정도로 많다.
- Freshness(새로움에 대한 적응의 문제?) : 유튜브에는 1초당 a lot of hour(꽤 많은 시간)에 해당하는 영상들이 업로드된다. 추천 시스템은 새롭게 추가되는 영상 컨텐츠에도 잘 작용해야 한다. 이 문제는 exploitation/exploration의 관점으로 바라 볼 수도 있다.
- Noise(불완전한 데이터) : 유튜브에서 유저의 historical 행동은 예측하기가 매우 어려운데 이는 data 자체의 sparsity나 관측이 불가능한 외적 요인들에서 기인한다. 유저에 대한 explicit한 데이터들보다 implicit한 데이터들을 보통 얻는다.<br>
 이러한 트레이닝 데이터셋의 특징들에 robust(단단한) 모델을 만들어야 한다.
>> 데이터 규모가 너무 크고, 유저들에 의한 새로운 데이터들이 계속 생성되며, 생성되는 데이터들 조차 dense하지 않고 sparse한 문제 등을 안고 있다.
Tensorflow 기반의 딥러닝으로 문제 해결하고자 했음. 기존의 MF 기반 방식들에 비해 해당 연구 시기(2016년 전후)에는 딥 러닝 기반의 추천 시스템에 대한 연구가 부족했음. 기존에 CF가 DNN이나 auto-encoder에 의해 모방된 연구들이 있긴 했다.

## System overview
후보군 선정, 랭킹 크게 두 부분으로 이뤄져 있다. 후보군 선정에서는 유튜브의 유저 활동 히스토리를 인풋으로 하고 매우 큰 corpus에서 수백개 정도의 작은 subset을 생성했다. 이 후보군들은 일반적으로 유저와 매우 높은 precision 연관이 있도록 돼있다. 후보군 선정 네트워크는 CF를 통한 broad한 개인화를 제공한다. 유저간의 유사도는 시청한 비디오들의 ID, search query token, demographics 등의 coarse한 feature들로 표현 될 수 있다. 최선의 추천을 나타내기 위해선 후보들 간의 상대적 중요성을 구별해서 보여줄 필요가 있다. 유저와 비디오들을 표현하는 풍부한 피쳐 셋들을 사용한 목적 함수를 따라서 각각의 비디오에 점수를 매김으로써, 랭킹을 매길 수 있다. 가장 높은 점수의 비디오들이 점수에 따라 유저에게 노출됨. 알고리즘 발전 과정에서는 precision, recall, ranking loss 등의 offline metrics들을 사용했으나, 최종적인 알고리즘의 효율 결정에 있어서는 실제 실험에서의 A/B 테스트를 사용했다. 미세한 CTR변동이나 시청 시간과 같은 다양한 측정들을 사용했는데, offline측정이 실제 A/B 테스트 결과와 항상 높은 상관관계가 있는 것이 아니므로 중요한 부분이라고 볼 수 있다.

## Candidate generation
유저의 이전 시청 정보가 임베딩 된 네트워크를 활용한 얕은 NN의 반복을 통해 기존의 MF 방법론을 모방하였음. 이러한 관점에서 보면 이 방법론은 non-linear generalization of factorization techniques으로 볼 수 있음.
