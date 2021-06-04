
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "mediapipe/examples/desktop/graph_runner.hpp"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

#include <iostream>

ABSL_FLAG(std::string, graph_config, "",
          "Name of file containing text format CalculatorGraphConfig proto.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  std::string graph_config = absl::GetFlag(FLAGS_graph_config);

  MP::GraphRunner runner;
  if(!runner.init(graph_config))
  {
    LOG(ERROR) << "Failed to init the graph";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  capture.open(0);

  if(!capture.isOpened()){
    LOG(ERROR) << "Cannot open camera";
    return EXIT_FAILURE;
  }

  std::cout << capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')) << std::endl;
  std::cout << capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280) << std::endl;
  std::cout << capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720) << std::endl;
  std::cout << capture.set(cv::CAP_PROP_FPS, 30) << std::endl;

  bool grab_frames = true;

  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
    }
    cv::Mat camera_frame;

    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::flip(camera_frame, camera_frame, 1);

    runner.process(camera_frame);

    if (!runner.ouput().empty())
      cv::imshow("MP", runner.ouput());

    // Press any key to exit.
    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255)
      grab_frames = false;
  }

  return EXIT_SUCCESS;
}