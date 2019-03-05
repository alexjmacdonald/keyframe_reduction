#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <algorithm> // sort
#include <cstdint>
#include <iostream> // std::cout, cerr
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

  // Copy frame indices, if provided, then sort & unique
  std::vector<uint32_t> v =
    vm.count("indices") ? vm["indices"].as<std::vector<uint32_t>>()
                        : std::vector<uint32_t>();
  {
    using namespace std;
    sort(v.begin(), v.end());
    auto last = unique(v.begin(), v.end());
    v.erase(last, v.end());
  }
  const std::vector<uint32_t> &frame_indices = v;

  // frame_idx starts at the first video,
  // and proceeds continuously across subsequent vids.
  // let
  //   vid 1 := 20 frames
  //   vid 2 := 30 frames
  //   vid 3 := 40 frames
  // then
  //   frame_idx 0 -> 90
  //   for frame_idx = 55, we're at vid 2 : frame 35
  uint32_t frame_idx = 0;
  // if frame indices are provided,
  // use them to seek frame_idx.
  auto frame_it     = frame_indices.begin();
  auto frame_it_end = frame_indices.end();
  // if no indices are provided,
  // select _all_ frames

  std::vector<std::string> results{};
  constexpr uint32_t ROWS     = 322;
  constexpr uint32_t COLS     = 240;
  constexpr uint32_t GRIDSIZE = 4;
  reduce_frames<ROWS,COLS,GRIDSIZE>(results, paths, frame_idx, frame_indices, frame_it, frame_it_end);

  return 0;
}

template <uint32_t ROWS, uint32_t COLS, uint32_t GRIDSIZE, typename ResultContainer, typename PathCollection, typename FrameIndexIterator, typename FrameIndexSequence>
bool reduce_frames(ResultContainer &result, PathCollection &paths, uint32_t &frame_idx, FrameIndexSequence &frame_indices, FrameIndexIterator &frame_it, FrameIndexIterator &frame_it_end) {
  for (const auto &path : paths) {
    using namespace cv;

    Mat image = cv::imread(path, 1);

    // Buffer will contain grayscale pixels row * col
    // TODO uint8_t should be arbitrary precision
    uint8_t* grayscale_buf = new uint8_t[ ROWS * COLS ];

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(path);

    if (!video.isOpened()) {
      std::cerr << "Error opening video stream or file" << std::endl;
      return false;
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
      // assuming this.  TODO live asserts are bad, should do something to ensure continuity in preprocessing
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
      constexpr uint32_t pixel_ct = COLS * ROWS;
      const uint8_t *pixel = frame.ptr<uint8_t>(0);
      // TODO do you ever get videos with INT_MAX pixels per frame?
      for (uint32_t pixel_idx = 0; pixel_idx < pixel_ct; pixel_idx++) {

        const uint32_t x = pixel_idx / COLS;
        const uint32_t y = pixel_idx % COLS;

        // Default OpenCV format is BGR
        const auto *pk_pixel = (pixel + pixel_idx * 3);
        uint8_t b = static_cast<uint8_t>(*(pk_pixel + 0));
        uint8_t g = static_cast<uint8_t>(*(pk_pixel + 1));
        uint8_t r = static_cast<uint8_t>(*(pk_pixel + 2));

        // convert pixel to grayscale
        grayscale(r, g, b);

        constexpr auto ROW_WIDTH = COLS;
        constexpr auto COL_WIDTH = ROWS;
        //std::cerr << pixel_idx << "\t" << maploc<GRIDSIZE, ROW_WIDTH, COL_WIDTH>(y,x) << "/" << (frame.rows * frame.cols) << std::endl;
        grayscale_buf[maploc<GRIDSIZE, ROW_WIDTH, COL_WIDTH>(y,x)] = grayscale(r,g,b);
        //assert((maploc<3, 9, 9>(0, 0) == 0));
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
  return true;
}
