Building:
--------

    $ cmake -DCMAKE_BUILD_TYPE=Debug -H. -Bbuild
    $ cmake --build build -- -j3
    $ ./build/keyframe_reduction --files ./test_inputs/whatever.mp4  # see usage

Compilation notes:
------------------

  ROWS, COLS, and GRIDSIZE are #defined constants within main.hpp.
  Change those to fit your video, compile & execute.

  If you're segfaulting, it may because you got the params wrong.


Usage:
------

    # Prints help:

      $ keyframe_reduction

      $ keyframe_reduction --help

    # Process all frames of videos:

      $ keyframe_reduction --files a.mp4

      $ keyframe_reduction --files a.mp4 b.mp4 c.mp4  # treats [a,b,c] as 1 video

      $ keyframe_reduction --files b.mp4 c.mp4 a.mp4  # treats [b,c,a] as 1 video

    # Process selected frames of videos:  (using ffprobe, etc. to select keyframes)
    # I tested with mp4; should have broad support from OpenCV

      $ ffprobe -select_streams v -i x.mp4 -print_format json -show_entries frame=pict_type,coded_picture_number  # copied from response email

      $ keyframe_reduction --files a.mp4 --indices 1 2 600

      $ keyframe_reduction --files a.mp4 --indices 60000 1 1 2 600  # indices are sorted & deduped

    # Diagnostic prints to STDERR, csv output prints to STDOUT.
    # Redirect accordingly.

      $ keyframe_reduction --files a.mp4 --indices 1 2 600 > results.csv


Design notes:
-------------

This utility was written with the following performance/design goals:

  - respect cache locality
  - minimize branches and branch mispredictions (i.e. optimize pipelining)
  - allow the loops to be easily parallelized (horizontal threading)
  - reasonably separate processing phases (vertical threading)

This includes extensive use of compile-time constants.

Bugs:
-----
  - timestamp assumes consistent FPS across video chunks
  - did I printf format the timestamps?
  - video chunks must match ROWS/COLS expectations, or violently segfault

Overall TODOs:
--------------
  - would be nice if opencv could detect keyframes and I didn't have to pass in ffprobe
  - move unit testing to cmake
  - do feature-level testing (rigging an input will take more time than I have)
  - cmake should define grid size, row width, col width, instead of header defs
  - benchmark
  - cmake should output executable name reflecting the above (e.g., keyframe\_reduction\_320\_240\_8 for 320x240 with grid size 8)
  - parallelization
  - idk why homebrew isn't installing gdb/valgrind.  only 1 mem allocation, doubtful this leaks
  - write a shell script wrapper for ffprobe / redirect to csv
  - was pretty lazy about mixing boost/stdlib

How to /properly/ find depth & channel ct?  this seems suboptimal
some info on format defs:

    https://docs.opencv.org/4.0.1/d1/d1b/group__core__hal__interface.html
    https://codeyarns.com/2015/08/27/depth-and-type-of-matrix-in-opencv/

    Mat first_frame;
    video >> first_frame;
    std::cout << first_frame.channels() << std::endl;
    std::cout << (video.get(CAP_PROP_FORMAT)) << std::endl;  // CV_8U etc, see above docs
