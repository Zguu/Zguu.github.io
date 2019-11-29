---
title: " [머신러닝] Clustering Algorithms"
tags: MachineLearning Clustering Algorithm K-means
---
# Mean-Shift Clustering
$\ $ 다른 클러스터링 알고리즘들에 비해 간단하고, 많은 장점이 있는 Mean-Shift Clustering에 대해 알아보자. 이 알고리즘은 kernel density estimation(KDE)의 개념에 입각한 알고리즘인데, KDE는 데이터 셋의 기저에 놓여있는 분포에 대하여 추정하는 것이다.
!['Img'](https://imgur.com/Ycp0gHX.png)
위와 같은 데이터 분포가 있다고 할때, 확률 분포 surface에 대하여 가우시안 커널을 활용하여 표현해보자.
!['Img1'](https://imgur.com/iaN7huN.png)
!['Img2'](https://imgur.com/cRr5uPs.png)
가우시안 커널을 bandwidth = 2로 적용하여 얻은 KDE 표면은 위와 같다. 각각 surface plot, contour plot을 보여준다.

## Mean Shift
$\ $Mean shift는 KDE 표면에서 가장 가까운 곳에 있는 peak(산봉우리)를 오르는 아이디어를 그대로 가져왔다. bandwidth 값을 어떻게 잡느냐에 따라서, 표면 형태가 아래와 같이 달라질 것이다. K-means에서 우리가 클러스터의 수를 직접 정해주는 것과 비슷하게, 이 알고리즘에서도 bandwidth 값 설정을 통해 결과로 나오는 클러스터의 수를 결정할 수 있다.

!['Img3'](https://imgur.com/lSBMC3C.gif)
!['Img4'](https://imgur.com/IubEWgq.gif)

위의 예시에서 보이듯이, bandwidth 크기에 따라서 최종 클러스터 갯수가 달라지게 된다. hill climbing의 일종으로 이해할 수 있다.<br>
$\ $sliding-window-based 알고리즘의 관점에서 보면, mean shift는 원형 모양의 window가 데이터 포인트들의 밀도가 높은 지점으로 이동하며 클러스터 센터를 바꿔가는 것으로 이해할 수 있다. 아래의 이미지는 이 shift 과정의 개념을 잘 보여준다.
!['Img5'](https://imgur.com/85CUMzE.png)
