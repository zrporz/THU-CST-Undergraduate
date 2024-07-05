# 运行说明

第一次运行时，需要执行
```
python predict.py --preprocess
```
以进行数据预处理

进行训练及预测时，需要选定使用的算法，包括XGBoost(XGB)，随机森林(RF)，全连接神经网络(MLP)
即
```
python predict.py --method XGB
或
python predict.py --method RF
或
python predict.py --method MLP
```

其中随机森林的训练时间较长，可能需要约十分钟

网页版报告：https://zrp21.notion.site/Report-85de73b712494213beed0cd1f07f1e1a?pvs=4