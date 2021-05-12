## Experiment

1. download and extract the [avazu dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data), then put `train.csv` into `experiment/`
2. select data in the first 3 days:

```
cd experiment/
python 0_avazu_data_proc.py
```

3. run the models:

`python 5avazu_test_l2_0.py`

`python 6avazu_test_l2_1e-4.py`

