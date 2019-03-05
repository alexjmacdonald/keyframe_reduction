#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <typeinfo> // for typeid debugging
#include <cstdint> // for typeid debugging

#include <assert.h>

// global TODOs:
// - replace short/int with sized types u8 u32 etc
// - move testing to cmake
// - cmake should determine grid size, row width, col width

template <uint32_t GRID, uint32_t _ROW_WIDTH, uint32_t _COL_WIDTH>
uint32_t maploc(uint32_t r, uint32_t c) {
  // TODO validation strategy,
  // right now we assume the caller isn't going to send OOB coords
  // this is probably appropriate (avoid branching) but should circle back

  // If the grid size doesn't evenly divide,
  // we simply ignore the margin.
  constexpr auto ROW_WIDTH = _ROW_WIDTH - (_ROW_WIDTH % GRID);
  constexpr auto COL_WIDTH = _COL_WIDTH - (_COL_WIDTH % GRID);
  // Consider a 3x3 grid over 9x9 pixels:
  /*
    * a b c  x x x  x x x
    * d e f  x x x  x x x
    * g h i  x x x  x x x
    *
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    *
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    * x x x  x x x  x x x
    */
  // For a regular linear traversal:
  // [ a b c . . . . . . d e f ... ]
  // the members of a cell aren't consecutive.
  //
  // For efficiency's sake (maintaining cache coherency & avoiding needless reiteration)
  // we make one pass across the image, copying pixels to a new buffer, and mapping
  // them to new locations in the buffer so that we get:
  // [ a b c d e f g h i . . . . . . ... ]
  //
  // The formula for this looks like:
  //
  /*
    *  (using ordinal ranks, not 0-based indexing)
    *
    *  Presume we map the contents of each grid cell
    *  to its grid's new index in the new buffer:
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  1  1   2  2  2   3  3  3
    *  |  1  1  1   2  2  2   3  3  3
    *  |  1  1  1   2  2  2   3  3  3
    *
    *  |  4  4  4   5  5  5   6  6  6
    *  |  4  4  4   5  5  5   6  6  6
    *  |  4  4  4   5  5  5   6  6  6
    *
    *  and then we account for the grid size
    *  (so, 1+0, 1+9, 1+9+9...)
    *  so that each cell maps to the grid's actual mem index:
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  1  1   10 10 10  19 19 19
    *     1  1  1   10 10 10  19 19 19
    *  |  1  1  1   10 10 10  19 19 19
    *
    *  |  28 28 28  37 37 37  46 46 46
    *     28 28 28  37 37 37  46 46 46
    *  |  28 28 28  37 37 37  46 46 46
    *
    *  Pretty close!
    *
    *  Finally, we use addition to compose our grid cell's index
    *  with the cell element's index within the grid cell:
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  0  1  2   0  1  2   0  1  2
    *     3  4  5   3  4  5   3  4  5
    *  |  6  7  8   6  7  8   6  7  8
    *
    *  |  0  1  2   0  1  2   0  1  2
    *     3  4  5   3  4  5   3  4  5
    *  |  6  7  8   6  7  8   6  7  8
    *
    *  Regular piecewise matrix addition.
    *
    *  Therefore...
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  2  3   4  5  6   7  8  9
    *     10 11 12  13 14 15  16 17 18
    *  |  19 20 21  22 23 24  25 26 27
    *
    *  |  28 29 30  31 32 33  34 35 36
    *     37 38 39  40 41 42  43 44 45
    *  |  46 47 48  49 50 51  52 53 54
    *
    *     becomes
    *
    *    x- - - - - - - - - - - - - -
    *  y
    *  |  1  2  3   10 11 12  19 20 21
    *     4  5  6   13 14 15  22 23 24
    *  |  7  8  9   16 17 18  25 26 27
    *
    *  |  28 29 30  37 38 39  46 47 48
    *     31 32 33  40 41 42  49 50 51
    *  |  34 35 36  43 44 45  52 53 54
    */

    const uint32_t cells_above        = (r/GRID) * (COL_WIDTH / GRID);
    const uint32_t cells_before       = c / GRID;
    const uint32_t starting_grid_idx  = (GRID*GRID) * (cells_above + cells_before);
    const uint32_t within_grid_idx    = (r%GRID)*GRID + c%GRID;

    return (starting_grid_idx + within_grid_idx);
}

void tests() {
  /*
   * ordinal positions (subtr 1 to get zero-based idx)
   *    x- - - - - - - - - - - - - -
   *  y
   *  |  1  2  3   10 11 12  19 20 21
   *     4  5  6   13 14 15  22 23 24
   *  |  7  8  9   16 17 18  25 26 27
   *
   *  |  28 29 30  37 38 39  46 47 48
   *     31 32 33  40 41 42  49 50 51
   *  |  34 35 36  43 44 45  52 53 54
   */

  /* for printf debugging the asserts below
    for (int i = 0; i < 54; i++) {
      int r = i/9;
      int c = i%9;
      auto x = (maploc<3, 9, 9>(r, c));
      std::cout << ">> " << i << "\t" << "\t" << r << "/" << c << ":\t" << x << std::endl;
    }
  */

  // basics
  assert( (maploc<3, 9, 9>(0, 0) == 0)  );
  assert( (maploc<3, 9, 9>(1, 0) == 3)  );
  assert( (maploc<3, 9, 9>(1, 1) == 4)  );
  assert( (maploc<3, 9, 9>(5, 0) == 33) );
  assert( (maploc<3, 9, 9>(2, 8) == 26) );

  // when grid doesn't evenly divide matrix
  assert( (maploc<3, 9, 10>(0, 0) == 0)  );
  assert( (maploc<3, 9, 10>(1, 0) == 3)  );
  assert( (maploc<3, 9, 10>(1, 1) == 4)  );
  assert( (maploc<3, 9, 10>(5, 0) == 33) );
  assert( (maploc<3, 9, 10>(2, 8) == 26) );
}

uint8_t grayscale(const uint8_t r, const uint8_t g, const uint8_t b) {
  // luminosity = 0.299R + 0.587G + 0.114B
  return (299*r + 587*g + 114*b) / 1000;
}

int main(int32_t argc, char** argv) {

  tests();

  // Usage / handle CLI
  boost::program_options::variables_map vm;
  {
    using namespace boost::program_options;

    options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("files",  value<std::vector<std::string>>() -> multitoken() -> required(), "list of file paths")
      // TODO optionally ingest list of keyframe indices
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

    // Allocate new buffer

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
      video >> frame;

      // A cv::Mat (frame) is 'continuous' when its internal representation is equivalent to one long array.
      // We process the underlying array directly, assuming this.
      assert(frame.isContinuous());

      if (frame.empty())
        break;

      // How to /properly/ find depth & channel ct?  this seems suboptimal
      // some info on format defs:
      // https://docs.opencv.org/4.0.1/d1/d1b/group__core__hal__interface.html
      // https://codeyarns.com/2015/08/27/depth-and-type-of-matrix-in-opencv/
      /*
      Mat first_frame;
      video >> first_frame;
      std::cout << first_frame.channels() << std::endl;
      std::cout << (video.get(CAP_PROP_FORMAT)) << std::endl;  // CV_8U etc, see above docs
      */

      // https://stackoverflow.com/questions/8184053/accessing-elements-of-a-cvmat-with-atfloati-j-is-it-x-y-or-row-col/42327350#42327350
      // alleges row-major storage
      // mat.at(i, j) = mat.at(row, col) = mat.at(y, x)
      //
      // assuming the iterator runs consecutively across array respecting cache
      // [0, 0, 0;     Row major layout
      //  2, 0, 0;  => 0 0 0 2 0 0 0 0 0
      //  0, 0, 0]     ^ --->
      const uint32_t row_ct            = frame.rows;
      const uint32_t col_ct            = frame.cols;
      const uint32_t pixel_ct          = col_ct * row_ct;
      const unsigned char* pixel  = frame.ptr<unsigned char>(0);
      // TODO do you ever get videos with INT_MAX pixels?
      for(int pixel_idx = 0; pixel_idx < pixel_ct; pixel_idx++) {

        const uint32_t x = pixel_idx / col_ct;
        const uint32_t y = pixel_idx % col_ct;

        // Default OpenCV format is BGR
        uint8_t b = static_cast<uint8_t>(*(pixel + pixel_idx*3 + 0));
        uint8_t g = static_cast<uint8_t>(*(pixel + pixel_idx*3 + 1));
        uint8_t r = static_cast<uint8_t>(*(pixel + pixel_idx*3 + 2));

        // convert pixel to grayscale
        grayscale(r,g,b);
        // use maploc() to copy to new buffer
        // determine median from buffer chunks
        //   depending on size of N,
        //   quickselect or insertion sort?
        //   TODO benchmark and then use GRID SIZE in template
        //   to determine
        // map buffer chunks to median
        // print to stdout, achieve csv write via redirection

      }
    }

    video.release();
  }

  return 0;
}
