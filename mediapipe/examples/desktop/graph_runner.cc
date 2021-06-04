// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "mediapipe/examples/desktop/graph_runner.hpp"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

#include <iostream>

namespace MP {

GraphRunner::GraphRunner()
{
  m_graph = std::shared_ptr<mediapipe::CalculatorGraph>(new mediapipe::CalculatorGraph());
}

GraphRunner::~GraphRunner()
{
}

bool GraphRunner::init(const std::string& graphConfig)
{
  m_graphConfig = graphConfig;
  std::string content;
  mediapipe::file::GetContents(graphConfig, &content);
  LOG(INFO) << "Get calculator graph config contents: " << content;
  mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(content);
  LOG(INFO) << "Initialize the calculator graph.";
  if(m_graph->Initialize(config).ok()){

    m_graph->ObserveOutputStream("multi_face_landmarks", [](const mediapipe::Packet& packet) {
      //LOG(INFO) << "multi_face_landmarks : Packet " << packet.DebugString();
      const auto& multi_face_landmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
      for (int face_index = 0; face_index < multi_face_landmarks.size(); ++face_index) {
        //LOG(INFO) << face_index; 
        const auto& landmarks = multi_face_landmarks[face_index];
        for (int i = 0; i < landmarks.landmark_size(); ++i) {
          LOG(INFO) << i << ": " <<landmarks.landmark(i).x() << "," << landmarks.landmark(i).y() << "," << landmarks.landmark(i).z();
        }
      }
      return absl::OkStatus();
    });

    absl::StatusOr<mediapipe::OutputStreamPoller> statusOrPoller = m_graph->AddOutputStreamPoller(kOutputStream);

    if(statusOrPoller.ok()) {
      m_poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(statusOrPoller.value()));
      LOG(INFO) << "Starting graph ...";
      return m_graph->StartRun({}).ok();
    }
  }
  return false;
}

void GraphRunner::process(cv::Mat input)
{
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, input.cols, input.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  m_input = mediapipe::formats::MatView(input_frame.get());
  input.copyTo(m_input);

  // Send image packet into the graph.
  mediapipe::Timestamp ts = mediapipe::Timestamp((double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6);
  absl::Status status = m_graph->AddPacketToInputStream( kInputStream, mediapipe::Adopt(input_frame.release()).At(ts));

  if(!status.ok()) {
    LOG(INFO) << "Add packet to input failed code(" << status.code() << ")";
    return;
  }

  mediapipe::Packet packet;
  if (!m_poller->Next(&packet))
  {
    LOG(INFO) << "Poller next failed";
    return;
  }

  auto& output_frame = packet.Get<mediapipe::ImageFrame>();
  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
  cv::cvtColor(output_frame_mat, m_output, cv::COLOR_RGB2BGR);
}

}
