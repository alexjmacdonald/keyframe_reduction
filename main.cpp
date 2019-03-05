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
#include <sstream>
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
  reduce_frames<VIDEO_ROWS,VIDEO_COLS,VIDEO_GRIDSIZE>(results, paths, frame_idx, frame_indices, frame_it, frame_it_end);

  for (auto line : results) {
    std::cout << line << std::endl;
  }

  return 0;
}

template <uint32_t RAW_ROWS, uint32_t RAW_COLS, uint32_t GRIDSIZE, typename ResultContainer, typename PathCollection, typename FrameIndexIterator, typename FrameIndexSequence>
bool reduce_frames(ResultContainer &result, PathCollection &paths, uint32_t &frame_idx, FrameIndexSequence &frame_indices, FrameIndexIterator &frame_it, FrameIndexIterator &frame_it_end) {
  for (const auto &path : paths) {
    using namespace cv;

    Mat image = cv::imread(path, 1);

    // If rows or cols are not an integral multiple of gridsize,
    // we ignore the excess.
    constexpr auto WASTE_ROWS = (RAW_ROWS % GRIDSIZE);
    constexpr auto WASTE_COLS = (RAW_COLS % GRIDSIZE);
    constexpr auto ROWS = RAW_ROWS - WASTE_ROWS;
    constexpr auto COLS = RAW_COLS - WASTE_COLS;

    // Buffer will contain grayscale pixels row * col
    // TODO uint8_t should be arbitrary precision
    constexpr uint32_t pixel_ct = COLS * ROWS;

    uint8_t* grayscale_buf = new uint8_t[ pixel_ct ];

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(path);

    const double fps = video.get(CAP_PROP_FPS);

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
      // assuming the iterator runs consecutively across array respecting cache locality
      // [0, 0, 0;     Row major layout
      //  2, 0, 0;  => 0 0 0 2 0 0 0 0 0
      //  0, 0, 0]     ^ --->
      //
      // Looks like openCV has its own stack overflow --
      // http://answers.opencv.org/question/38/what-is-the-most-effective-way-to-access-cvmat-elements-in-a-loop/
      // Code works for now, but TODO look into pitch/stride * channel ct as way to iterate

      // Little trick here to advance the pixel ptr past the waste columns.
      // At the beginning of the loop, we assume our pointer is at waste:
      //
      //   [     good     ][  waste  ]
      // -----------------^
      //   [     good     ][  waste  ]
      //   [     good     ][  waste  ]
      //   [     good     ][  waste  ]
      //   [          waste          ]
      //
      // so we seek 1 waste-width, wrapping us to the next row:
      //
      //   [     good     ][  waste  ]
      //   [     good     ][  waste  ]
      // --^
      //   [     good     ][  waste  ]
      //   [     good     ][  waste  ]
      //   [          waste          ]
      //
      // To avoid an additional condition, we prime this invariant so that
      // the first loop iteration sets us back to 0:
      //
      //   [ entirely unrelated mem  ]
      // -----------------^
      //   [     good     ][  waste  ]
      //   [     good     ][  waste  ]
      //   [     good     ][  waste  ]
      //   [     good     ][  waste  ]
      //   [          waste          ]
      //
      // Tailing waste rows are handled by simply not iterating over them.
      const uint8_t *pixel = frame.ptr<uint8_t>(0) - WASTE_COLS;

      for (uint16_t row_idx = 0; row_idx < ROWS; row_idx++) {
        pixel += (WASTE_COLS * 3); // BGR -> 3 channels per pixel, see above todo about channel determination
        for (uint16_t col_idx = 0; col_idx < COLS; col_idx++) {

          // Mapping function (where, in the new buffer, the pixel goes)
          constexpr auto mapper = &maploc<GRIDSIZE, COLS, ROWS>;

          const auto loc = mapper(col_idx, row_idx);

          // Default OpenCV format is BGR
          uint8_t b = static_cast<uint8_t>(*(pixel + 0));
          uint8_t g = static_cast<uint8_t>(*(pixel + 1));
          uint8_t r = static_cast<uint8_t>(*(pixel + 2));

          constexpr auto ROW_WIDTH = COLS;
          constexpr auto COL_WIDTH = ROWS;
          grayscale_buf[loc] = grayscale(r,g,b);

          pixel += 3;
        }
      }

      // Sort each cell enough that arr[floor(n/2)] is the median.
      // (for small N, selection sort seems efficient?)
      // TODO this requires actual benchmarking for various N
      constexpr auto cell_size  = GRIDSIZE*GRIDSIZE;
      constexpr auto cell_count = (ROWS * COLS) / cell_size;
      {
        auto p = grayscale_buf;
        for (uint32_t i = 0; i < cell_count; i++) {
          // TODO write a real sort algo here
          std::sort(p, p + cell_size);
          p += cell_size;
        }
      }

      // TODO benchmark
      // is it better to separate or combine these loops?

      // Accumulate each floor(n/2)'th member of each cell,
      // which will be the median.
      std::stringstream ss;
      {
        constexpr auto median_idx = (cell_size / 2);
        auto p = grayscale_buf + median_idx;
        const double timestamp = ((fps * frame_idx) / 1000.0);
        ss << timestamp << "," << static_cast<uint32_t>(*p);
        for (uint32_t i = 1; i < cell_count; i++) {
          ss << ",";
          ss << static_cast<uint32_t>(*p);
          p += cell_size;
        }
      }

      result.push_back(ss.str());
    }
    video.release();
    delete[] grayscale_buf;
  } // end iterating over paths
  return true;
}
