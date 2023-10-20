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

#include <include/structure_layout.h>

namespace PaddleOCR {

void StructureLayoutRecognizer::Run(cv::Mat img,
                                    std::vector<StructurePredictResult> &result,
                                    std::vector<double> &times) {
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  // preprocess
  auto preprocess_start = std::chrono::steady_clock::now();

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat resize_img;
  this->resize_op_.Run(srcimg, resize_img, 800, 608);
  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());
  auto preprocess_end = std::chrono::steady_clock::now();
  preprocess_diff += preprocess_end - preprocess_start;

  // get shape of norm_img_batch
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
  std::vector<const char *> output_names;
  for (auto& output_name_ptr : output_names_ptr) {
    output_names.push_back(output_name_ptr.get());
  }

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(),
                                                            input.size(), input_shape.data(),
                                                            input_shape.size());

  auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
                                     input_names.size(), output_names.data(), output_names.size());

  // Get output tensor
  std::vector<std::vector<float>> out_tensor_list;
  std::vector<std::vector<int>> output_shape_list;

  for (int j = 0; j < output_names.size(); j++) {
    std::vector<int64_t> output_shape = output_tensors[j].GetTensorTypeAndShapeInfo().GetShape();
    int64_t output_count = std::accumulate(output_shape.begin(), output_shape.end(), 1, 
                                           std::multiplies<int64_t>());
    float *float_array = output_tensors[j].GetTensorMutableData<float>();
    std::vector<float> out_data(float_array, float_array + output_count);

    std::vector<int> int_output_shape;
    for (int i = 0; i < output_shape.size(); i++) {
      int_output_shape.push_back(output_shape[i]);
    }

    output_shape_list.push_back(int_output_shape);
    out_tensor_list.push_back(out_data);
  }                           
  auto inference_end = std::chrono::steady_clock::now();
  inference_diff += inference_end - inference_start;

  // postprocess
  auto postprocess_start = std::chrono::steady_clock::now();

  std::vector<int> bbox_num;
  int reg_max = 0;
  for (int i = 0; i < out_tensor_list.size(); i++) {
    if (i == this->post_processor_.fpn_stride_.size()) {
      reg_max = output_shape_list[i][2] / 4;
      break;
    }
  }
  std::vector<int> ori_shape = {srcimg.rows, srcimg.cols};
  std::vector<int> resize_shape = {resize_img.rows, resize_img.cols};
  this->post_processor_.Run(result, out_tensor_list, ori_shape, resize_shape,
                            reg_max);
  bbox_num.push_back(result.size());

  auto postprocess_end = std::chrono::steady_clock::now();
  postprocess_diff += postprocess_end - postprocess_start;
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
}

void StructureLayoutRecognizer::LoadModel(const std::string &model_dir) {
  std::string model_file = model_dir + "/inference.onnx";
  this->session = new Ort::Session(this->env, model_file.c_str(), this->sessionOptions);
  
  const size_t in_num = session->GetInputCount();
  Ort::AllocatorWithDefaultOptions allocator;
  // for (int i = 0; i < in_num; ++i) {
  //       auto name = session->GetInputNameAllocated(i, allocator);
  //       std::cout << "Input Name: " << name.get() << std::endl;
        
  //       auto type_info = session->GetInputTypeInfo(i);
  //       auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  //       auto input_node_dims = tensor_info.GetShape();
  //       printf("Input num_dims = %zu\n", input_node_dims.size());
  //       for (size_t j = 0; j < input_node_dims.size(); j++) {
  //         printf("Input dim[%zu] = %llu\n",j, input_node_dims[j]);
  //       }
  // }
}
} // namespace PaddleOCR
