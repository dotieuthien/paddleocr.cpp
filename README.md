Just clone PaddleOCR cpp_infer, and using OnnxRuntime instead of Paddle

### 1. Export the inference model
Download PaddleOCR models, and directory structure is as follows
```
inference/
|-- det_db
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- rec_rcnn
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- cls
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- table
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- layout
|   |--inference.pdiparams
|   |--inference.pdmodel
```

Export to Onnx using paddle2onnx.
```
paddle2onnx --model_dir saved_inference_model \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams\
            --save_file inference.onnx \
            --enable_dev_version True
```
Onnx model structure.
```
inference/
|-- det_db
|   |--inference.onnx
|-- rec_rcnn
|   |--inference.onnx
|-- cls
|   |--inference.onnx
|-- table
|   |--inference.onnx
|-- layout
|   |--inference.onnx
```

### 2. Compile PaddleOCR
Still in cmake dev to build dependencies as static libs

### 3. Run the demo
##### 1. det+cls+rec：
```shell
./build/PaddleOcrOnnx 
    --det_model_dir=inference/det_db \
    --rec_model_dir=inference/rec_rcnn \
    --cls_model_dir=inference/cls \
    --image_dir=images/1.jpg \
    --use_angle_cls=true \
    --det=true \
    --rec=true \
    --cls=true \
```

##### 2. layout
```shell
./build/PaddleOcrOnnx 
    --layout_model_dir=inference/layout \
    --image_dir=images/1.png \
    --type=structure \
    --table=false \
    --layout=true \
    --det=false \
    --rec=false
```

### 4. Reference
[PaddleOCR cpp_infer](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7/deploy/cpp_infer): origin implementation of PaddleOCR cpp

[PaddleOCR + OnnxRuntime](https://github.com/RapidAI/RapidOcrOnnx/tree/61d7b434d2b773eb61dab85328240789f69b3ae0): the repo has no layout function
