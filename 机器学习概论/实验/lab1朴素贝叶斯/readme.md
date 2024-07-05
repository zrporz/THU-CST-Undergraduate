# README
###### 周韧平 2021010699 zrp21@mails.tsinghua.edu.cn

- `build_dataset.py`: 运行可以对原始数据进行处理，清洗好的数据存在 `output.json` 中
- `train_wordbag_add_feat.py`: 执行训练和测试的代码，默认随机数种子123，下面介绍其各个参数
    - `alpha`: 平滑系数
    - `feat_<feat_name>`: 赋值为True引入该特征
    - `ratio`: 样本量(取值0到1)
    - `weight`: 除了词袋模型外特征的权重系数
    - `predict_zeros`: 设为True后推测时考虑词袋模型中 $x_k^i = 0$ 
- `plot*.py`: 绘制report中图片的代码