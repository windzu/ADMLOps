#include "detector.h"
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

int main() {
  const char *device_name = "cuda";
  int device_id = 0;

  // get mmdeploy model path of faster r-cnn
  std::string mmdeploy_path = std::getenv("MMDEPLOY_DIR");
  std::string model_path = mmdeploy_path + "/work_dir";
  // use mmdetection demo image as an input image
  std::string mmdet_path = std::getenv("MMDETECTION_DIR");
  std::string image_path = mmdet_path + "/demo/demo.jpg";

  // create inference handle
  mm_handle_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path.c_str(), device_name,
                                            device_id, &detector);
  assert(status == MM_SUCCESS);

  // read image
  cv::Mat img = cv::imread(image_path);
  assert(img.data);

  // apply handle and get the inference result
  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};
  mm_detect_t *bboxes{};
  int *res_count{};
  status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
  assert(status == MM_SUCCESS);

  // deal with the result. Here we choose to visualize it
  for (int i = 0; i < *res_count; ++i) {
    const auto &box = bboxes[i].bbox;
    if (bboxes[i].score < 0.3) {
      continue;
    }
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom},
                  cv::Scalar{0, 255, 0});
  }

  cv::imwrite("output_detection.png", img);

  // destroy result buffer
  mmdeploy_detector_release_result(bboxes, res_count, 1);
  // destroy inference handle
  mmdeploy_detector_destroy(detector);

  return 0;
}