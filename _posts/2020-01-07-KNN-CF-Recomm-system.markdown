---
title: " [머신러닝] KNNBasic Collaborative Recommender system"
tags: MachineLearning Recommender System KNN KNNBasic
---
# CF의 종류
$\ $ Collaborative Filter (CF)는 협업 필터라고 번역할 수 있다. 이는 일반적으로 우리가 추천하고자 하는 영화나 상품들에 대한 특성들만을 이용해서 유사한 영화 또는 품목을 추천하는 Contents Based 방식과는 차이가 있다. 협업 필터는 다른 유저들이 해당 품목에 어떤 평가를 내렸는지, 구매 이력이 어떻게 되는지 등의 유저가 기록한 데이터 또한 사용하여 추천에 활용한다. 협업 필터에는 크게 memory-based 방식과 model-based 방식이 있는데, 왜 이렇게 크게 두 갈래로 나뉘어 지는지에 대해 잠시 짚고 넘어가보자.

(user-item 행렬 샘플 이미지 첨부)

메모리 베이스 방식은 기존에 유저들이 기록한 user-item 행렬에 있는 데이터를 활용해 지금까지 있었던 값들에서 가장 유사한 값을 찾아준다. 더 쉽게 얘기해보면, 나와 가장 비슷한 유저를 찾거나, 내가 좋아하는 영화와 가장 비슷한 영화를 찾는 방식이다.

모델 베이스 방식은 위의 방식과는 조금 더 다르게 예측의 영역을 넘본다. 현재 있는 데이터 행렬을 Latent factor를 활용해 decomposition을 해낸 후 다시 결합하여 sparse 파트들의 빈 값들을 찾아내는 MF 방식이 대표적인 방법이다. SVD, SVD++, funkSVD과 같은 방식들이 있다. 일반적으로는 행렬 분해에서 아이디어를 얻은 것들에서 시작한 영역인 듯 하다.

2000년대 초반에 실제 기업들이 메모리 베이스 CF 만으로도 큰 마케팅 효과를 보았다. 이번 포스트에서는 가장 기본적인이면서도 효과적인 user-based CF를 알아보도록한다.

(메모리 베이스 CF 들에서 최고 유사도 품목을 찾는 것 또한 알고리즘이 필요하다. KNN 등.. 이에 대한 설명을 추가하도록 한다.)

# KNNBasic CF 구현

surprise를 활용해 바로 구현하거나, 간단한 user-item 테이블 구성후에 scikit의 KNN을 활용하는 방법이 있다. surprise로 구현하면 너무 시시하기 때문에 그나마 조금 더 직관적인 이해를 도우면서 덜 시시한 후자의 방법을 택해서 설명을 해보도록 한다.

(다시 user-item 행렬 샘플 보여주며, user-base 와 item-base 설명)

이미지 하나 추가하면 좋을 듯 하다.

(파이썬 코드 쫘자락 및 설명)

> references:
  https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
  https://antilibrary.org/2086
  https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada
