# 使用手册

### 文件结构

|--data\
|------train\
|------test\
|--output  # 存放训练好的模型\
|--model  # 存放模型.py\
|--inference.py  # 模型使用\
|--OCRdataset.py  # 设计自己的dataset\
|--train.py  # 训练模型


### 模型训练

``` python
python train.py data_path='data'
```

**注意事项**：
1. data文件夹需按照格式放置，或自行修改dataloader中的参数
2. 模型训练测试相关代码在iteration方法中，需根据自身模型数据特性进行修改
3. 需要调整的参数基本都在parser中
4. 训练过程需要记录日志，包含每个epoch的loss，acc，lr及其他评价指标，模板给出了logging方式保存，如果需要其他的自行调试

### 模型使用

在文件Inference.py中，大致模板已给出，需要根据自己的模型任务做修改
