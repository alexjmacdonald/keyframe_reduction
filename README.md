Overall TODOs:
- move testing to cmake
- cmake should determine grid size, row width, col width

How to /properly/ find depth & channel ct?  this seems suboptimal
some info on format defs:
https://docs.opencv.org/4.0.1/d1/d1b/group\_\_core\_\_hal\_\_interface.html
https://codeyarns.com/2015/08/27/depth-and-type-of-matrix-in-opencv/
```
  Mat first_frame;
  video >> first_frame;
  std::cout << first_frame.channels() << std::endl;
  std::cout << (video.get(CAP_PROP_FORMAT)) << std::endl;  // CV_8U etc, see above docs
```
