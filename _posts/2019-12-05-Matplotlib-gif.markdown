---
title: " [파이썬 팁] Plot들 여러개를 움직이는 GIF로 저장하기"
tags: Python Matplotlib GIF
---

몇 줄 안되는 코드로 간단하게 GIF 움직이는 plot 저장하기
## 필요한 것들

ffmpeg는 homebrew로 따로 설치를 해준다. 아무것도 모르고 pip 패키징으로 설치했다가 안돌아가서 헤맸다...
- matplotlib
- celluliod (pip install)
- ffmpeg (homebrew)
- ffmpy (pip install)

패키징 업로드
```python
from matplotlib import pyplot as plt
from celluloid import Camera
import numpy as np
```

간단한 다항함수 (1차식부터 10차식까지)를 차수에 따라 달라지도록 설정한다.<br>
celluliod Camera() 함수 내에 우리의 그래프들이 그려질 figure를 넣어서 저장해주고, 각 그래프들이 새로 그려질 때마다 스크린샷을 찍는 느낌으로 camera.snap() 실행하여 저장해준다. 마지막으로 animate 함수로 지금까지 저장된 그림들을 연속적으로 보여준다. interval은 스냅샷들간의 전환 속도이다. ms 단위인듯 하다.  
```python
fig = plt.figure()
plt.ylim(0,1000)
camera = Camera(fig)
x = np.linspace(0,100,100)

for i in range(10):
    plt.title("Polynomial Function")
    plt.plot(x, x ** i)
    camera.snap()

animation = camera.animate(interval = 1000)
```

나는 쥬피터에서 돌렸는데, 해당 애니메이션으로 완성된 파일을 로컬로 저장하는 데에서 또 꽤 헤맸다. celluliod 모듈은 gif 저장 기능이 없어서, mp4로 1차 저장 이후, ffmpy 모듈을 이용해 해당 mp4를 바로 gif로 저장한다.
```python
animation.save("animation.mp4")
import ffmpy
ff = ffmpy.FFmpeg(
    inputs = {"animation.mp4" : None},
    outputs = {"animation6.gif" : None})

ff.run()
```
<center><img src="https://imgur.com/zdvSxCB.gif" width="60%" height="60%"></center>

**완성된 gif파일. 앞으로 유용하게 써먹을 듯 하다. 끗
