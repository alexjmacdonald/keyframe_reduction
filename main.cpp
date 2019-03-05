#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <typeinfo> // for typeid debugging
#include <vector>

#include <assert.h>

#include "main.hpp"

// Updates frame_idx in place.
// Returns true if frame_idx is still below frame_cap and
// within the collection's list of frames (if provided).
template<typename Ordinal, typename Iterator>
bool seek(bool use_container, Iterator &it, Iterator &end, Ordinal &frame_idx, Ordinal &frame_cap) {
  if (use_container) {
    if (it == end) return false;
    frame_idx = *(it++);
  } else {
    frame_idx++;
  }
  return frame_idx < frame_cap;
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
      std::cerr << ">> " << i << "\t" << "\t" << r << "/" << c << ":\t" << x <<
    std::endl;
    }
  */

  // basics
  assert((maploc<3, 9, 9>(0, 0) == 0));
  assert((maploc<3, 9, 9>(1, 0) == 3));
  assert((maploc<3, 9, 9>(1, 1) == 4));
  assert((maploc<3, 9, 9>(5, 0) == 33));
  assert((maploc<3, 9, 9>(2, 8) == 26));

  // when grid doesn't evenly divide matrix
  assert((maploc<3, 9, 10>(0, 0) == 0));
  assert((maploc<3, 9, 10>(1, 0) == 3));
  assert((maploc<3, 9, 10>(1, 1) == 4));
  assert((maploc<3, 9, 10>(5, 0) == 33));
  assert((maploc<3, 9, 10>(2, 8) == 26));
}

uint8_t grayscale(const uint8_t r, const uint8_t g, const uint8_t b) {
  // luminosity = 0.299R + 0.587G + 0.114B
  return (299 * r + 587 * g + 114 * b) / 1000;
}

int main(int32_t argc, char **argv) {

  tests();

  // Usage / handle CLI
  boost::program_options::variables_map vm;
  {
    using namespace boost::program_options;

    options_description desc{"Options"};
    desc.add_options()
        ("help,h",  "Help screen")
        ("files",
          value<std::vector<std::string>>()->multitoken()->required(),
          "list of file paths")
        ("indices",
          value<std::vector<uint32_t>>()->multitoken(),
          "frame indices to parse");

    store(parse_command_line(argc, argv, desc), vm);

    try {
      notify(vm);

      if (vm.count("help")) {
        std::cerr << desc << std::endl;
        return 0;
      }

    } catch (required_option &e) {
      std::cerr << desc << std::endl;
      return 1;
    }
  }

  // Obtain & validate file paths
  const auto &paths = vm["files"].as<std::vector<std::string>>();
  {
    std::cerr << "Proceeding with file list:" << std::endl;
    for (const auto &path : paths) {
      std::cerr << "  " << path << std::endl;

      if (!boost::filesystem::exists(path)) {
        std::cerr << "    \\_ "
                  << "Couldn't ensure file at that path exists!" << std::endl;
      }
    }

    bool files_all_present =
        boost::algorithm::all<decltype(paths),
                              bool(const boost::filesystem::path &)>(
            paths, boost::filesystem::exists);

    if (!files_all_present) {
      std::cerr << "Halting for missing files." << std::endl;
      return 1;
    }
  }

  const std::vector<uint32_t> &frame_indices =
      vm.count("indices") ? vm["indices"].as<std::vector<uint32_t>>()
                          : std::vector<uint32_t>();

  // frame_idx starts at the first video,
  // and proceeds continuously across subsequent vids.
  // vid 1 := 20 frames
  // vid 2 := 30 frames
  // vid 3 := 40 frames
  // frame_idx 0 -> 90
  uint32_t frame_idx = 0;
  // if frame indices are provided,
  // use them to seek frame_idx.
  auto frame_it     = frame_indices.begin();
  auto frame_it_end = frame_indices.end();
  // if no indices are provided,
  // select _all_ frames

  for (const auto &path : paths) {
    using namespace cv;

    Mat image = cv::imread(path, 1);

    // Buffer will contain grayscale pixels row * col
    // TODO uint8_t should be arbitrary precision
    uint8_t* grayscale_buf = new uint8_t[ image.rows * image.cols ];

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(path);

    if (!video.isOpened()) {
      std::cerr << "Error opening video stream or file" << std::endl;
      return 1;
    }

    uint32_t frame_ct = frame_idx + video.get(CAP_PROP_FRAME_COUNT);
    while (seek(frame_indices.size(), frame_it, frame_it_end, frame_idx, frame_ct)) {
      std::cerr << "Processing frame #" << frame_idx << std::endl;

      if (frame_indices.size()) {
        video.set(CAP_PROP_POS_FRAMES, frame_idx);
      }

      Mat frame;
      video >> frame;

      // A cv::Mat (frame) is 'continuous' when its internal representation is
      // equivalent to one long array. We process the underlying array directly,
      // assuming this.
      assert(frame.isContinuous());

      if (frame.empty())
        break;

      // https://stackoverflow.com/questions/8184053/accessing-elements-of-a-cvmat-with-atfloati-j-is-it-x-y-or-row-col/42327350#42327350
      // alleges row-major storage
      // mat.at(i, j) = mat.at(row, col) = mat.at(y, x)
      //
      // assuming the iterator runs consecutively across array respecting cache
      // [0, 0, 0;     Row major layout
      //  2, 0, 0;  => 0 0 0 2 0 0 0 0 0
      //  0, 0, 0]     ^ --->
      const uint32_t row_ct = frame.rows;
      const uint32_t col_ct = frame.cols;
      const uint32_t pixel_ct = col_ct * row_ct;
      const uint8_t *pixel = frame.ptr<uint8_t>(0);
      // TODO do you ever get videos with INT_MAX pixels per frame?
      for (uint32_t pixel_idx = 0; pixel_idx < pixel_ct; pixel_idx++) {

        const uint32_t x = pixel_idx / col_ct;
        const uint32_t y = pixel_idx % col_ct;

        // Default OpenCV format is BGR
        const auto *pk_pixel = (pixel + pixel_idx * 3);
        uint8_t b = static_cast<uint8_t>(*(pk_pixel + 0));
        uint8_t g = static_cast<uint8_t>(*(pk_pixel + 1));
        uint8_t r = static_cast<uint8_t>(*(pk_pixel + 2));

        // convert pixel to grayscale
        grayscale(r, g, b);
        // use maploc() to copy to new buffer
        // determine median from buffer chunks
        //   depending on size of N,
        //   selection sort, stop halfway (4 iterations is nothing)
        //   quickselect or insertion sort?
        //   TODO benchmark and then use GRID SIZE in template
        //   to determine
        // map buffer chunks to median
        // print to stdout, achieve csv write via redirection
      }
    }
    video.release();
    delete[] grayscale_buf;
  } // end iterating over paths

  return 0;
}
