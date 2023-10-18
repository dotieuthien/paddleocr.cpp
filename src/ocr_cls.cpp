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

#include <include/ocr_cls.h>

namespace PaddleOCR {

void Classifier::Run(std::vector<cv::Mat> img_list,
                     std::vector<int> &cls_labels,
                     std::vector<float> &cls_scores,
                     std::vector<double> &times) {
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  int img_num = img_list.size();
  std::vector<int> cls_image_shape = {3, 48, 192};
  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->cls_batch_num_) {
    auto preprocess_start = std::chrono::steady_clock::now();
    int end_img_no = std::min(img_num, beg_img_no + this->cls_batch_num_);
    int batch_num = end_img_no - beg_img_no;

    // preprocess
    std::vector<cv::Mat> norm_img_batch;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      cv::Mat srcimg;
      img_list[ino].copyTo(srcimg);
      cv::Mat resize_img;
      this->resize_op_.Run(srcimg, resize_img, this->use_tensorrt_,
                           cls_image_shape);

      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      if (resize_img.cols < cls_image_shape[2]) {
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                           cls_image_shape[2] - resize_img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      }
      norm_img_batch.push_back(resize_img);
    }
    std::vector<float> input(batch_num * cls_image_shape[0] *
                                 cls_image_shape[1] * cls_image_shape[2],
                             0.0f);
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    preprocess_diff += preprocess_end - preprocess_start;

    // get shape of norm_img_batch
    std::array<int64_t, 4> input_shape{cls_batch_num_, 3, 48, 192};

    // inference with onnx
    Ort::AllocatorWithDefaultOptions allocator;

    // get input names ptr
    const size_t in_num = session->GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    input_names_ptr.reserve(in_num);
    std::vector<int64_t> input_node_dims;
    for (size_t i = 0; i < in_num; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_names_ptr.push_back(std::move(input_name));
    }

    // get output name ptr
    const size_t out_num = session->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    output_names_ptr.reserve(out_num);
    std::vector<int64_t> output_node_dims;
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

    auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
                                      input_names.size(), output_names.data(), output_names.size());
    std::vector<int64_t> predict_shape = output_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    // for (size_t j = 0; j < predict_shape.size(); j++) {
    //         printf("Predicted shape dim[%zu] = %llu\n",j, predict_shape[j]);
    // }
    
    int64_t output_count = std::accumulate(predict_shape.begin(), predict_shape.end(), 1, 
                                       std::multiplies<int64_t>());

    // output tensor value                                   
    float *float_array = output_tensor.front().GetTensorMutableData<float>();
    std::vector<float> predict_batch(float_array, float_array + output_count);

    auto inference_end = std::chrono::steady_clock::now();
    inference_diff += inference_end - inference_start;

    // postprocess
    auto postprocess_start = std::chrono::steady_clock::now();
    for (int batch_idx = 0; batch_idx < predict_shape[0]; batch_idx++) {
      int label = int(
          Utility::argmax(&predict_batch[batch_idx * predict_shape[1]],
                          &predict_batch[(batch_idx + 1) * predict_shape[1]]));
      float score = float(*std::max_element(
          &predict_batch[batch_idx * predict_shape[1]],
          &predict_batch[(batch_idx + 1) * predict_shape[1]]));
      cls_labels[beg_img_no + batch_idx] = label;
      cls_scores[beg_img_no + batch_idx] = score;
    }
    auto postprocess_end = std::chrono::steady_clock::now();
    postprocess_diff += postprocess_end - postprocess_start;
  }
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
}

void Classifier::LoadModel(const std::string &model_dir) {
  std::cout << "Load model classification" << std::endl;
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
