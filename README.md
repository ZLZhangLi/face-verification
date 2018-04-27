# DeepID-I-implementation
Re-implementation of DeepID on Caffe framework.<br> 
* This repository is just a naive implementation of DeepID, its accuracy seems `hard` to reproduce(According to some callbacks).
* More Info about face recognition on Mobile Devices or Embedded Board could be easily get on GitHub.
* 1. [Face recognition](https://github.com/ageitgey/face_recognition)
* 2. [DeepID](https://github.com/stdcoutzyx/DeepID_FaceClassify)
* ...

#### Our Work:
*Training Set: CASIA WebFace `cropped to 144x144` Gray (Washed)<br>
Validation Set: LFW `cropped to 128x128` Gray (6K Pairs)<br>
Accuracy @LFW:  `91.9667%`<br>
*:It won't be necessary to set Validation Set on WebFace. You may get a higher precision on LFW with a larger training data set.

[Network Structure](http://ethereon.github.io/netscope/#/gist/5b46e36eb7c92c95798afde3bec4e780)<br>

#### Pre-trained Caffemodel:
CHN : [Baidu Pan](https://pan.baidu.com/s/1pKUVgPd)<br>
Other country: [One Drive](https://1drv.ms/u/s!Av1MQK8mV3J8bP9is3GBhpyCySw)<br>

#### Related Blog:
[CSDN](http://blog.csdn.net/baidu_24281959/article/details/53462168)<br>

#### [Washed WebFace DataSet](http://pan.baidu.com/s/1hrKpbm8)<br>
