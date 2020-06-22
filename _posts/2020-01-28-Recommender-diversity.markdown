---
title: "[머신러닝] 추천 시스템 Diversity 측정 정리"
tags: MachineLearning Recommender Diversity
---
<center><img src="https://imgur.com/5A87q6B.png" width="90%" height="90%"></center>

# 추천 시스템에서의 다양성
$ \$`MoiveLens`와 같은 데이터셋에 추천 알고리즘을 적용한 추천 시스템을 구축할 때에, 얼마나 이 시스템이 유저의 평점을 정확히 예측하는 지는 매우 중요한 평가 척도이다. 하지만, 이러한 예측 정확도 못지 않게, 실제 많은 e-commerce 환경에서는 추천 시스템의 다양성 또한 고려해야 한다. 너무 편협하게 소수의 컨텐츠만 반복적으로 유저에게 추천하면, 유저 입장에서는 쉽게 지루해질 수 있다. 또한 시스템의 관점에서는 새로운 상품을 추천하지 못하고, 유저들이 충분히 매력을 느낄만한 아직 유저들에 의해 발굴되지 못한 상품들에 대해서도 유저에게 제시해 줄 수 없다. e-commerce 에서 빈번하게 발생하는 `Long-tail` 효과에 빠질 위험이 있다. 이런 롱테일 효과에서 꼬리 부분에 해당하는 소수 상품들을 추출해서 적절한 유저에게 추천하는 것은 전체 세일즈의 life-time에도 영향을 끼칠 것이다. 따라서, 추천 시스템에서의 예측 정확도 만큼이나, 다양성 또한 충분히 고려해돼야 할 metrics 중에 하나라고 볼 수 있다.
<center><img src="https://imgur.com/wT01nXY.png" width="60%" height="60%"><img src="https://imgur.com/xGfR9P8.png" width="60%" height="60%"></center>

## 다양성 측정을 위한 다양한 측정들
$\ $추천 시스템이 얼마나 다양한 추천 결과를 냈는 지를 추천하는 metrics들은 여러가지가 있다. 유저들에게 영화를 추천하는 상황이라고 할 때, 추천 시스템 적용 이후 얼마나 많은 영화 (unique number of movies)들을 유저들이 보고 있는 지를 볼 수도 있고, 유저들이 영화들에 매긴 평점 값의 분산의 추이(Global variance of ratings)를 확인해 볼 수도 있다. 이와 약간 다르게, 각각 유저 단위로 해당 유저들이 남긴 영화 평점의 분산 값들의 평균(Mean user-based variance of ratings)을 볼 수도 있다. 조금 더 나아가, `Shannon Entropy`, `Cosine Diversity` 를 사용해 볼 수도 있다.
> Shannon Entropy : $$H = -\sum_{i=1}^n p_i\cdot log(p_i); p_i = \frac{occ(movie_i)}{count(ratings)}$$<br>
  Cosine Diversity : Cosine similarity 의 반대 개념 ( simply, $$1 - \sum Similarity(U_i, U_j$$ ))

추가적인 다른 다양성 지표들은 아래의 Table에서 확인해 볼 수 있다.
<center><img src="https://imgur.com/LenFc60.png" width="100%" height="100%"></center>

> References<br>
https://www.benefitfocus.com/blogs/benefitfocus/what-the-employee-benefits-business-can-learn-from-the-retail-industry<br>
Diversity in recommender systems - A survey<br>
Diversity Measurement of Recommender Systems under Different User Choice Models
https://twitter.com/Doc_Hillz/status/908164008548683781/photo/1
