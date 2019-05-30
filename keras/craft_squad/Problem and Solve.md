
### Problem 1 : Lambda layer can not save

"""
Traceback (most recent call last):
  File "/home/apollo/Link to Projects/Holy-QA/solution_keras/keras_run.py", line 79, in <module>
    K_train.train_baseline()
  File "/media/apollo/6bc96a82-9055-48ec-a649-6bb58bdf64ba/Projects/Holy-QA/solution_keras/train_keras.py", line 240, in train_baseline
    model.save(self.model_name)
  File "/opt/anaconda3/lib/python3.6/site-packages/keras/engine/topology.py", line 2556, in save
    save_model(self, filepath, overwrite, include_optimizer)
  File "/opt/anaconda3/lib/python3.6/site-packages/keras/models.py", line 108, in save_model
    'config': model.get_config()
  File "/opt/anaconda3/lib/python3.6/site-packages/keras/engine/topology.py", line 2397, in get_config
    return copy.deepcopy(config)
  File "/opt/anaconda3/lib/python3.6/copy.py", line 240, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/opt/anaconda3/lib/python3.6/copy.py", line 169, in deepcopy
    rv = reductor(4)
TypeError: can't pickle _thread.lock objects
"""
+ https://stackoverflow.com/questions/49425056/keras-lambda-layer-and-variables-typeerror-cant-pickle-thread-lock-objects

### Problem 2 predict index error
+ 输出的 beg index 和 end index 存在 一些问题
+ 比如 beg index 大于 end index
+ 措施, 找到 begin idx 和 end idx 的 top K (5)
    + 对找到的 begin idx 和 end idx 进行 约束
    + 参考 SQuAD 的部分论文
    + 参考 solution_tf_SQuAD
    + 将predict 的结果保存下来
+ 直接以 pointer network 等工作 查找答案
+ 整理获取答案的方法

### model.evaluate 与　model predict 的区别

### 测试代码有问题
+ 保存predict 的结果
+ 参考 solution_tensorflow_squad
