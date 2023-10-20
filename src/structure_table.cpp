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

#include <include/structure_table.h>

namespace PaddleOCR {

void StructureTableRecognizer::Run(
    std::vector<cv::Mat> img_list,
    std::vector<std::vector<std::string>> &structure_html_tags,
    std::vector<float> &structure_scores,
    std::vector<std::vector<std::vector<int>>> &structure_boxes,
    std::vector<double> &times) {
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  int img_num = img_list.size();
  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->table_batch_num_) {
    // preprocess
    auto preprocess_start = std::chrono::steady_clock::now();
    int end_img_no = std::min(img_num, beg_img_no + this->table_batch_num_);
    int batch_num = end_img_no - beg_img_no;
    std::vector<cv::Mat> norm_img_batch;
    std::vector<int> width_list;
    std::vector<int> height_list;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      cv::Mat srcimg;
      img_list[ino].copyTo(srcimg);
      cv::Mat resize_img;
      cv::Mat pad_img;
      this->resize_op_.Run(srcimg, resize_img, this->table_max_len_);
      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      this->pad_op_.Run(resize_img, pad_img, this->table_max_len_);
      norm_img_batch.push_back(pad_img);
      width_list.push_back(srcimg.cols);
      height_list.push_back(srcimg.rows);
    }

    std::vector<float> input(
        batch_num * 3 * this->table_max_len_ * this->table_max_len_, 0.0f);
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    preprocess_diff += preprocess_end - preprocess_start;

    // get shape of norm_img_batch
    std::array<int64_t, 4> input_shape{batch_num, 3, this->table_max_len_, this->table_max_len_};

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
    std::vector<int64_t> predict_shape0_ = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t out_num0 = std::accumulate(predict_shape0_.begin(), predict_shape0_.end(), 1, 
                                       std::multiplies<int64_t>());                                  
    float *float_array0 = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> loc_preds(float_array0, float_array0 + out_num0);

    std::vector<int64_t> predict_shape1_ = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    int64_t out_num1 = std::accumulate(predict_shape1_.begin(), predict_shape1_.end(), 1, 
                                       std::multiplies<int64_t>());                                  
    float *float_array1 = output_tensors[1].GetTensorMutableData<float>();
    std::vector<float> structure_probs(float_array1, float_array1 + out_num1);

    auto inference_end = std::chrono::steady_clock::now();
    inference_diff += inference_end - inference_start;

    std::vector<int> predict_shape0;
    for (int i = 0; i < predict_shape0_.size(); i++) {
      predict_shape0.push_back(predict_shape0_[i]);
    }

    std::vector<int> predict_shape1;
    for (int i = 0; i < predict_shape1_.size(); i++) {
      predict_shape1.push_back(predict_shape1_[i]);
    }

    // postprocess
    auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<std::vector<std::string>> structure_html_tag_batch;
    std::vector<float> structure_score_batch;
    std::vector<std::vector<std::vector<int>>> structure_boxes_batch;
    this->post_processor_.Run(loc_preds, structure_probs, structure_score_batch,
                              predict_shape0, predict_shape1,
                              structure_html_tag_batch, structure_boxes_batch,
                              width_list, height_list);

    for (int m = 0; m < predict_shape0[0]; m++) {

      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                         "<table>");
      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                         "<body>");
      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                         "<html>");
      structure_html_tag_batch[m].push_back("</table>");
      structure_html_tag_batch[m].push_back("</body>");
      structure_html_tag_batch[m].push_back("</html>");

      structure_html_tags.push_back(structure_html_tag_batch[m]);
      structure_scores.push_back(structure_score_batch[m]);
      structure_boxes.push_back(structure_boxes_batch[m]);
    }
    auto postprocess_end = std::chrono::steady_clock::now();
    postprocess_diff += postprocess_end - postprocess_start;
    times.push_back(double(preprocess_diff.count() * 1000));
    times.push_back(double(inference_diff.count() * 1000));
    times.push_back(double(postprocess_diff.count() * 1000));
  }
}

void StructureTableRecognizer::LoadModel(const std::string &model_dir) {
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
