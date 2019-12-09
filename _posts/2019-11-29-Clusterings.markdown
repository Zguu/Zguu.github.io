---
title: " [머신러닝] Clustering Algorithms"
tags: MachineLearning Clustering Algorithm K-means
---
# Mean-Shift Clustering
$\ $ 다른 클러스터링 알고리즘들에 비해 간단하고, 많은 장점이 있는 Mean-Shift Clustering에 대해 알아보자. 이 알고리즘은 kernel density estimation(KDE)의 개념에 입각한 알고리즘인데, KDE는 데이터 셋의 기저에 놓여있는 분포에 대하여 추정하는 것이다.
<center><img src="https://imgur.com/Ycp0gHX.png" width="60%" height="60%"></center>
위와 같은 데이터 분포가 있다고 할때, 확률 분포 surface에 대하여 가우시안 커널을 활용하여 표현해보자.
<center><img src="https://imgur.com/iaN7huN.png" width="60%" height="60%"></center>
<center><img src="https://imgur.com/cRr5uPs.png" width="60%" height="60%"></center>
가우시안 커널을 bandwidth = 2로 적용하여 얻은 KDE 표면은 위와 같다. 각각 surface plot, contour plot을 보여준다.

## Mean Shift
$\ $Mean shift는 KDE 표면에서 가장 가까운 곳에 있는 peak(산봉우리)를 오르는 아이디어를 그대로 가져왔다. bandwidth 값을 어떻게 잡느냐에 따라서, 표면 형태가 아래와 같이 달라질 것이다. K-means에서 우리가 클러스터의 수를 직접 정해주는 것과 비슷하게, 이 알고리즘에서도 bandwidth 값 설정을 통해 결과로 나오는 클러스터의 수를 결정할 수 있다.
<center><img src="https://imgur.com/lSBMC3C.gif" width="60%" height="60%"></center>
<center><img src="https://imgur.com/IubEWgq.gif" width="60%" height="60%"></center>
위의 예시에서 보이듯이, bandwidth 크기에 따라서 최종 클러스터 갯수가 달라지게 된다. hill climbing의 일종으로 이해할 수 있다.<br>
$\ $sliding-window-based 알고리즘의 관점에서 보면, mean shift는 원형 모양의 window가 데이터 포인트들의 밀도가 높은 지점으로 이동하며 클러스터 센터를 바꿔가는 것으로 이해할 수 있다. 아래의 이미지는 이 shift 과정의 개념을 잘 보여준다.
<center><img src="https://imgur.com/85CUMzE.gif" width="60%" height="60%"></center>
1. 임의의 시작점에서 반지름 r의 원을 그린다. (bandwidth)
2. 해당 원안에 포함된 점들의 중점을 찾는다.
3. 해당 중점을 중심으로 다시 반지름 r의 원을 그린다.
4. 2,3 번의 움직임을 계속해서 반복한다.
위 과정들은 밀도가 높은 점으로 자연스럽게 원이 움직이도록 유도한다.
>크기가 일정한 태풍이 움직인다고 생각해보자, 이 태풍은 현재 위치에서, 습도가 가장 높은 쪽으로 계속해서 움직인다고 생각해보자 (태풍의 눈이 습도가 가장 높은 곳으로 이동한다). 동시다발적으로 태풍이 여러 지역에서 발생했을 때, 해당 태풍들이 최종적으로 위치한 곳을 살펴보면 당연히! 습도가 해당 지역들에서 가장 높은 곳일 것이다. 아래 그림을 참고하자

<center><img src="https://imgur.com/B8BE0A5.gif" width="60%" height="60%"></center>
Mean shift 알고리즘은 K-means 와는 다르게, 미리 cluster의 갯수를 정해줄 필요는 없다. 비슷하게 band-width를 우리가 지정하긴 하지만 이 값은 cluster의 값을 우리가 사전에 정하는 것과는 분명 다르다.
## DBSCAN (Density-Based Spatial Clustering of Application with Noise)
$\ $DBSCAN은 mean shift와 비슷하게 데이터들의 밀도를 활용한 클러스터링 방법으로, 해당 공간에 데이터들이 주어졌을 때, 가장 가까이 위치하고 있는 점들을 그룹화한다.

1. 다른 클러스터링 방법들과 비슷하게, 특정 점에서 알고리즘을 시작한다.
2. **특정 점** 을 기준으로 거리가 $$\epsilon$$ 내에 있는 모든 점을 이웃으로 인지하는데, 인지되는 이웃이 우리가 미리 잡아 놓은 수 minPoints 이상이면, **특정 점** 은 core points로 구분한다.
3. **어떤 점** 이 core point와 $\epsilon$거리에는 있지만, 막상 이 점 주변에는 minPoints 이상의 점이 없는 경우, 이 **어떤 점** 은 cluster 내에는 있지만, core points는 아니고 directly reachable인 점으로 구분한다.
4. core points도 아니고, directly reachable points도 아닌 경우에는, outliers 또는 noise points로 구분한다.

아래의 그림을 보면 조금 더 쉽게 이해된다.
<center><img src="https://imgur.com/7kl2Owj.png" width="60%" height="60%"></center>
