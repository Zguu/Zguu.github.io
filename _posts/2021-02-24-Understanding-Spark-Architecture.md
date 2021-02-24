---
title: " [Spark Feature] Spark Architecture Explained"
tags: Spark BigData
---

# Spark
스파크는 빅데이터 처리에서 대세로 자리잡은 클러스터 컴퓨팅 프레임 워크입니다. 스파크는 하둡과 비교했을 때, in memory 에서는 100배 이상, on disk 에 비해서는 10배 까지 빠른 데이터 처리 속도를 보여줍니다. 이러한 장점을 지닌 스파크의 전체적인 아키텍쳐에 대해서 알아봅시다.

## Spark & its Features

스파크의 주된 특징은 **in-memory cluster computing** 인데, 이는 프로세스 속도를 증가시킵니다. 전체 클러스터들에 대하여, implicit **data parallelism and fault tolerance** 에 대한 프로그래밍 작성을 위해 인터페이스를 제공합니다. batch applications, iteractive algorithms, interactive queries, and straming 과 같은 다양한 범위의 작업들을 커버할 수 있도록 디자인 됐습니다.

Features of Apache Spark:

![Features of Spark](https://i.imgur.com/JWmgOIp.png)

1. Speed
   위에서 언급했듯이, 하둡 MapReduce 에 비해 100배 빠른 속도를 보여줍니다. 이는 잘 통제된 partitioning을 통해 이뤄집니다.

2. Powerful Caching
   Simple programming layer가 caching & disk 지속성을 제공합니다.

3. Deployment
   **Mesos, Hadroop via Yarn or Spark's own cluster manager** 에 의해 작동됩니다.

4. Real-Time
   **in-memory computation** 이기 때문에, 실시간 연산 & 낮은 latnecy 를 보여줍니다.
5. Polyglot
   low level 뿐 아니라 high level API를 제공합니다. 이러한 API들은 Java, Scala, python, R 코드에서 작성될 수 있도록 돕기 때문에 매우 확장성이 높습니다.

> Release vs Deploy vs Distribute 모두 "배포하다?" 의 의미일까?
> 일반적으로 Release는 같은 제품을 새롭게 만드는 것.
> Deploy는 프로그램 등을 서버와 같은 기기에 설치하여 작동가능하도록 만드는 것
> Distribute는 제품을 사용자들이 사용할 수 있도록 서비스 등을 제공하는 의미
> Ex) FaceBook 버전 x가 새롭게 Release 되었고, 이를 서버에 deploy 하여 사용자들이 사용할 수 있도록 distribute 하였다. (출처 https://opentutorials.org/course/1724/9836)

## Spark Architecture Overviw

스파크는 well-defined layered architecture 를 갖추고 있는데, 모든 스파크 구성요소들과 레이어들은 loosely coupled 돼 있습니다. 이는 더욱 다양한 확장 프로그램 또는 외부 라이브러리들과의 통합에 유리하다고 이해하면 될 것 같습니다. 아파치 스파크의 아키텍셔는 기본적으로 두 개의 메인 abstraction에 기초하고 있습니다.

- Resilient Distributed Dataset(RDD)
- DIrected Acyclic Graph(DAG)

스파크 아키텍쳐의 더욱 자세한 부분을 들여다보기 전에, Spark Ecosystem, RDD와 같은 기본적 개념들에 대해 한 번 짚고 넘어가겠습니다. 이러한 기초를 이해하고 넘어가는 것이 스파크에 대한 인사이트를 얻는데에 도움이 될 것입니다.

![Spark Architecture](https://i.imgur.com/wVuSMmd.png)

## Spark Ecosystem

스파크는 아래와 같은 ecosystem을 갖추고 있습니다. SparkSQL, SparkStreaming, MLlib, GraphX and CoreAPI component 등이 있습니다.

![Spark Eco-System](https://i.imgur.com/BfAmhLm.png)

1. Spark Core
  스파크 코어는 대용량 병렬, 분산 데이터 처리의 기본 엔진입니다. 또한 추가적인 라이브러리들은 이 코어에 설치되며, 다양한 워크로드를 허용합니다. 이는 SQL, 머신러닝, 스트리밍과 같은 것들입니다. 코어는 메모리 관리와 fault 회복, 스케쥴링, 분산 및 job 감시에 대한 책임을 지게 됩니다.
2. Spark Streaming
  실시간 데이터를 처리하는 데에 필요한 구성 요소입니다.
3. Spark SQL
  스파크 SQL 는 SQL 언어 또는 Hive Query 언어를 통해 데이터를 Query하게 합니다. RDBMS 에 친숙한 유저들에게 Spark SQL은 매우 친숙하게 사용될 수 있다는 것을 의미합니다.
4. GraphX
  GraphX는 graph와 graph-parallel 연산을 위한 Spark API입니다. 그러므로, 이 API는 스파크의 RDD를 Resilient Distributed Property Graph로 확장시킵니다.
5. MLlib (Machine Learning)
  Mllib는 머신러닝을 위한 라이브러리입니다. scikit-learn이나 tensorflow로 구현할 수 있는 ML의 일부를 편리하게 제공합니다.


## Resilient Distributed Dataset(RDD)
RDD는 스파크 어플리케이션들의 building block에 해당합니다. RDD는 다음과 같은 의미를 내포합니다.

- Resilient : Fault tolerant 한 특성 & rebuiling data on failure. 즉, 데이터에 오류가 생겼을 때 이에 대한 수정이 가능하며, 오류를 바로잡을 수 있습니다.
- Distributed : 여러 대의 cluter 컴퓨팅 인스턴스에 데이터들이 분산돼있습니다.
- Dataset : partition으로 나뉘어진 data들의 집합. 예를 들어, date로 partition 돼 있는 경우, 대용량의 데이터들을 날짜 별로 조회할 수 있습니다. 물론 파티션된 각 데이터셋은 value들을 포함한 이해하기 쉬운 데이터 형태입니다.

기본적으로 RDD 형태들은 immutable하며, lazy transformation을 따릅니다.

![RDD](https://i.imgur.com/JGP1FWl.png)

> lazy transformation은 대용량 데이터를 처리할 떄에 최적화된 변형 루트를 찾도록 합니다. 중간 데이터셋들이 계속해서 발생해서 GC 비용을 발생하는 것은 최소화하면서, 최종 데이터 처리형태로 가장 빠르게 도달하도록 합니다.

RDD 는 key를 기반으로 chunk들로 쪼개집니다. 복제된 chunk 조각들이 존재하기 때문에, spark는 손실된 데이터나 오류 데이터들을 회복시킬 수 있습니다.

RDD의 분산 환경에 대해서 이야기를 해봅시다. RDD에 있는 각각의 데이터셋들은 logical partition으로 분할되며, 이는 각각의 다른 클러스터 노드에서 연산될 수 있습니다. 이러한 특성 때문에, data transformation 또는 연산들이 각각의 data 조각에 대해 병렬로 이뤄질 수 있습니다.

![RDD2](https://i.imgur.com/tBjDhHP.png)
