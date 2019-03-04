#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <typeinfo> // for typeid debugging

int main(int argc, char** argv) {

  // Usage / handle CLI
  boost::program_options::variables_map vm;
  {
    using namespace boost::program_options;

    options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("files",  value<std::vector<std::string>>() -> multitoken() -> required(), "list of file paths")
      ;

    store(parse_command_line(argc, argv, desc), vm);

    try {
      notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
      }

    } catch(required_option& e) {
      std::cout << desc << std::endl;
      return 1;
    }
  }

  // Obtain & validate file paths
  auto paths = vm["files"].as<std::vector<std::string>>();
  {
    bool halt = false;
    std::cerr << "Proceeding with file list:" << std::endl;
    for (auto it = paths.begin() ; it != paths.end(); ++it) {
      const auto &path = *it;
      std::cerr << "  " << path << std::endl;

      if (!boost::filesystem::exists(path) ) {
        std::cerr << "    \\_ " << "Couldn't ensure file at that path exists!" << std::endl;
        halt = true;
      }
    }

    if (halt) {
      std::cerr << "Halting for missing files." << std::endl;
      return 1;
    }
  }

  for (auto it = paths.begin(); it != paths.end(); it++) {
    using namespace cv;

    const auto &path = *it;
    Mat image = cv::imread( path, 1 );

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(path);

    if (!video.isOpened()) {
      std::cout << "Error opening video stream or file" << std::endl;
      return 1;
    }

    for (int frame_idx = 0; frame_idx < video.get(CAP_PROP_FRAME_COUNT); frame_idx++) {
      // seek frame_idx according to keyframe list, if provided
      // TBD

      Mat frame;
      Mat grey_fun_frame;
      video >> frame;
      grey_fun_frame.create(frame.size(), frame.type());

      // A cv::Mat (frame) is 'continuous' when its internal representation is equivalent to one long array.
      // We process the underlying array directly, assuming this.
      assert(frame.isContinuous());

      if (frame.empty())
        break;

      // TODO how to /properly/ find depth & channel ct?  this seems suboptimal
      // some info on format defs:
      // https://docs.opencv.org/4.0.1/d1/d1b/group__core__hal__interface.html
      // https://codeyarns.com/2015/08/27/depth-and-type-of-matrix-in-opencv/
      /*
      Mat first_frame;
      video >> first_frame;
      std::cout << first_frame.channels() << std::endl;
      std::cout << (video.get(CAP_PROP_FORMAT)) << std::endl;  // CV_8U etc, see above docs
      */

      // Default OpenCV format is BGR

      // https://stackoverflow.com/questions/8184053/accessing-elements-of-a-cvmat-with-atfloati-j-is-it-x-y-or-row-col/42327350#42327350
      // alleges row-major storage
      // mat.at(i, j) = mat.at(row, col) = mat.at(y, x)
      //
      // assuming the iterator runs consecutively across array respecting cache
      // [0, 0, 0;     Row major layout
      //  2, 0, 0;  => 0 0 0 2 0 0 0 0 0
      //  0, 0, 0]     ^ --->
      const int row_ct            = frame.rows;
      const int col_ct            = frame.cols;
      const int pixel_ct          = col_ct * row_ct;
      const unsigned char* pixel  = frame.ptr<unsigned char>(0);
      // TODO do you ever get videos with INT_MAX pixels?
      for(int pixel_idx = 0; pixel_idx < pixel_ct; pixel_idx++) {
        const int x = pixel_idx / col_ct;
        const int y = pixel_idx % col_ct;
        // luminosity = 0.299R + 0.587G + 0.114B
        unsigned short b = static_cast<unsigned short>(*(pixel + pixel_idx*3 + 0));
        unsigned short g = static_cast<unsigned short>(*(pixel + pixel_idx*3 + 1));
        unsigned short r = static_cast<unsigned short>(*(pixel + pixel_idx*3 + 2));
        unsigned short grey = (299*r + 587*g + 114*b) / 1000;

        // TODO is an RGB triplicate format a constraint on imshow()
        // or are there properties on the matrix / flags for display I can tweak?
        grey_fun_frame.at<unsigned char>(x,y*3+0) = static_cast<unsigned short>(grey);
        grey_fun_frame.at<unsigned char>(x,y*3+1) = static_cast<unsigned short>(grey);
        grey_fun_frame.at<unsigned char>(x,y*3+2) = static_cast<unsigned short>(grey);
      }

      std::cout << frame_idx << std::endl;

      imshow( "Frame", frame_idx%20==0 ? frame : grey_fun_frame);

      char c=(char)waitKey(10);
      if(c==27)
        break;
    }

    // When everything done, release the video capture object
    video.release();
  }

  return 0;
}
