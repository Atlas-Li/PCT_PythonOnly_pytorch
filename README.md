# PCT: Point Cloud Transformer
This is a Pytorch implementation of PCT: Point Cloud Transformer.

Paper link: https://arxiv.org/pdf/2012.09688.pdf

### Requirements
python >= 3.7

pytorch >= 1.6

h5py

scikit-learn

### Dataset
Download the ModelNet40(http://modelnet.cs.princeton.edu/) with running "data.py" 
```shell script
python data.py
```


### Models
Get an accuracy of 93.2% on the ModelNet40(http://modelnet.cs.princeton.edu/) validation dataset

The path of the model is in ./checkpoints/best/models/model.t7

### Example training and testing
```shell script
# train
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001

# test
python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8

```
