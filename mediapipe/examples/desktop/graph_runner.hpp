

#include <string>

#include <opencv2/core/core.hpp>

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";

namespace mediapipe{
  class OutputStreamPoller;
  class CalculatorGraph;
}

namespace MP{

class GraphRunner
{
public:
  GraphRunner();
  virtual ~GraphRunner();

  bool init(const std::string& graphConfig);

  cv::Mat input() const { return m_input;}
  cv::Mat ouput() const { return m_output;}

  void process(cv::Mat input);

  void close();

private:
  std::string m_graphConfig;

  std::shared_ptr<mediapipe::CalculatorGraph> m_graph {nullptr};
  std::shared_ptr<mediapipe::OutputStreamPoller> m_poller {nullptr};

  cv::Mat m_input;
  cv::Mat m_output;
};
}