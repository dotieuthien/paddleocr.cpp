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

#include <include/ocr_rec.h>

namespace PaddleOCR {

void CRNNRecognizer::Run(std::vector<cv::Mat> img_list,
                         std::vector<std::string> &rec_texts,
                         std::vector<float> &rec_text_scores,
                         std::vector<double> &times) {
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  int img_num = img_list.size();
  std::vector<float> width_list;
  for (int i = 0; i < img_num; i++) {
    width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
  }
  std::vector<int> indices = Utility::argsort(width_list);

  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->rec_batch_num_) {
    auto preprocess_start = std::chrono::steady_clock::now();
    int end_img_no = std::min(img_num, beg_img_no + this->rec_batch_num_);
    int batch_num = end_img_no - beg_img_no;
    int imgH = this->rec_image_shape_[1];
    int imgW = this->rec_image_shape_[2];
    float max_wh_ratio = imgW * 1.0 / imgH;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      int h = img_list[indices[ino]].rows;
      int w = img_list[indices[ino]].cols;
      float wh_ratio = w * 1.0 / h;
      max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
    }

    int batch_width = imgW;
    std::vector<cv::Mat> norm_img_batch;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      cv::Mat srcimg;
      img_list[indices[ino]].copyTo(srcimg);
      cv::Mat resize_img;
      this->resize_op_.Run(srcimg, resize_img, max_wh_ratio,
                           this->use_tensorrt_, this->rec_image_shape_);
      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      norm_img_batch.push_back(resize_img);
      batch_width = std::max(resize_img.cols, batch_width);
    }

    std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    preprocess_diff += preprocess_end - preprocess_start;

    std::array<int64_t, 4> input_shape{batch_num, 3, imgH, batch_width};

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
    std::vector<int64_t> predict_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int64_t output_count = std::accumulate(predict_shape.begin(), predict_shape.end(), 1, 
                                       std::multiplies<int64_t>());

    // output tensor value                                   
    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> predict_batch(float_array, float_array + output_count);

    auto inference_end = std::chrono::steady_clock::now();
    inference_diff += inference_end - inference_start;

    // ctc decode
    auto postprocess_start = std::chrono::steady_clock::now();
    for (int m = 0; m < predict_shape[0]; m++) {
      std::string str_res;
      int argmax_idx;
      int last_index = 0;
      float score = 0.f;
      int count = 0;
      float max_value = 0.0f;

      for (int n = 0; n < predict_shape[1]; n++) {
        // get idx
        argmax_idx = int(Utility::argmax(
            &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
            &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
        // get score
        max_value = float(*std::max_element(
            &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
            &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
          score += max_value;
          count += 1;
          str_res += label_list_[argmax_idx];
        }
        last_index = argmax_idx;
      }
      score /= count;
      if (std::isnan(score)) {
        continue;
      }
      rec_texts[indices[beg_img_no + m]] = str_res;
      rec_text_scores[indices[beg_img_no + m]] = score;
    }
    auto postprocess_end = std::chrono::steady_clock::now();
    postprocess_diff += postprocess_end - postprocess_start;
  }
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
}

void CRNNRecognizer::LoadModel(const std::string &model_dir) {
  std::cout << "Load model recognition" << std::endl;
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
