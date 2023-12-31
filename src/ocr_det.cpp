// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <include/ocr_det.h>

namespace PaddleOCR {

void DBDetector::Run(cv::Mat &img,
                     std::vector<std::vector<std::vector<int>>> &boxes,
                     std::vector<double> &times) {
  float ratio_h{};
  float ratio_w{};

  cv::Mat srcimg;
  cv::Mat resize_img;
  img.copyTo(srcimg);

  auto preprocess_start = std::chrono::steady_clock::now();
  this->resize_op_.Run(img, resize_img, this->limit_type_,
                       this->limit_side_len_, ratio_h, ratio_w,
                       this->use_tensorrt_);

  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());
  auto preprocess_end = std::chrono::steady_clock::now();

  std::array<int64_t, 4> input_shape{1, 3, resize_img.rows, resize_img.cols};

  // inference with onnx
  Ort::AllocatorWithDefaultOptions allocator;

  // get input names ptr
  const size_t in_num = session->GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  input_names_ptr.reserve(in_num);
  for (size_t i = 0; i < in_num; i++) {
      auto input_name = session->GetInputNameAllocated(i, allocator);
      input_names_ptr.push_back(std::move(input_name));
  }

  // get output name ptr
  const size_t out_num = session->GetOutputCount();
  std::vector<Ort::AllocatedStringPtr> output_names_ptr;
  output_names_ptr.reserve(out_num);
  for (size_t i = 0; i < out_num; i++) {
      auto output_name = session->GetOutputNameAllocated(i, allocator);
      output_names_ptr.push_back(std::move(output_name));
  }

  // run
  auto inference_start = std::chrono::steady_clock::now();

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<const char *> input_names = {input_names_ptr.data()->get()};
  std::vector<const char *> output_names = {output_names_ptr.data()->get()};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(),
                                                            input.size(), input_shape.data(),
                                                            input_shape.size());

  auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
                                    input_names.size(), output_names.data(), output_names.size());
  std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int64_t output_count = std::accumulate(output_shape.begin(), output_shape.end(), 1, 
                                         std::multiplies<int64_t>());

  // output tensor value                                   
  float *float_array = output_tensors.front().GetTensorMutableData<float>();
  std::vector<float> out_data(float_array, float_array + output_count);

  auto inference_end = std::chrono::steady_clock::now();

  auto postprocess_start = std::chrono::steady_clock::now();
  int n2 = output_shape[2];
  int n3 = output_shape[3];
  int n = n2 * n3;

  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }

  cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
  cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

  const double threshold = this->det_db_thresh_ * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  if (this->use_dilation_) {
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, bit_map, dila_ele);
  }

  boxes = post_processor_.BoxesFromBitmap(
      pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
      this->det_db_score_mode_);

  boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff =
      preprocess_end - preprocess_start;
  times.push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times.push_back(double(inference_diff.count() * 1000));
  std::chrono::duration<float> postprocess_diff =
      postprocess_end - postprocess_start;
  times.push_back(double(postprocess_diff.count() * 1000));
}

void DBDetector::LoadModel(const std::string &model_dir) {
  std::cout << "Load model detection" << std::endl;
  std::string model_file = model_dir + "/inference.onnx";
  this->session = new Ort::Session(this->env, model_file.c_str(), this->sessionOptions);
  
  const size_t in_num = session->GetInputCount();
  Ort::AllocatorWithDefaultOptions allocator;
  for (int i = 0; i < in_num; ++i) {
    auto name = session->GetInputNameAllocated(i, allocator);
    std::cout << "Input Name: " << name.get() << std::endl;
    
    auto type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input_node_dims = tensor_info.GetShape();
    printf("Input num_dims = %zu\n", input_node_dims.size());
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      printf("Input dim[%zu] = %llu\n",j, input_node_dims[j]);
    }
  }
}

} // namespace PaddleOCR
