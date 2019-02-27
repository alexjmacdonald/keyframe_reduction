#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {

  // usage, cmd arg parsing
  boost::program_options::variables_map vm;
  {
    using namespace boost::program_options;

    // https://theboostcpplibraries.com/boost.program_options
    options_description desc{"Options"};
    desc.add_options()
      ("help,h",                                        "Help screen"        )
      ("files",   value<std::vector<std::string>>() -> multitoken() -> required(),   "list of file paths" );

    store(parse_command_line(argc, argv, desc), vm);

    try {
      notify(vm);

      if (vm.count("help"))
        std::cout << desc << std::endl;
      else if (vm.count("work"))
        std::cout << "do some keyframe stuff" << std::endl;

    } catch(required_option& e) {
      std::cout << desc << std::endl;
      return 1;
    }
  }

  auto paths = vm["files"].as<std::vector<std::string>>();
  {
    bool halt = false;
    std::cerr << "Proceeding with file list:" << std::endl;
    for (auto it = paths.begin() ; it != paths.end(); ++it) {
      const auto &filename = *it;
      std::cerr << "  " << filename << std::endl;

      if (!boost::filesystem::exists(filename) ) {
        std::cerr << "    \\_ " << "Couldn't access that file!" << std::endl;
        halt = true;
      }
    }

    if (halt) {
      std::cerr << "Halting for missing files." << std::endl;
      return 1;
    }

  }

  // auto image = cv::imread( xxx, 1 );

  return 0;
}
