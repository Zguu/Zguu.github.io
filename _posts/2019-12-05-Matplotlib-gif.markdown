---
title: " [파이썬 팁] Plot들 여러개를 움직이는 GIF로 저장하기"
tags: Bayesian Python Binomial prior posterior
---

그나마 간단하게 GIF로 움직이는 plot 저장하기
## 필요한 것들
matplotlib
celluliod
ffmpeg (brew)
ffmpy(pip install)

```python
from matplotlib import pyplot as plt
from celluloid import Camera
import numpy as np
```

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

```python
animation.save("animation.mp4")
import ffmpy
ff = ffmpy.FFmpeg(
    inputs = {"animation.mp4" : None},
    outputs = {"animation6.gif" : None})

ff.run()
```
<center><img src="https://imgur.com/zdvSxCB.gif" width="80%" height="60%"></center>
