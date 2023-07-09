# RoSAS

## Intro

Official implementation of the paper 
**"RoSAS: Deep Semi-supervised Anomaly Detection with Contamination-resilient Continuous Supervision"** 
published in *Information Processing & Management*.  (to appear)


RoSAS is a semi-supervised anomaly detection method that can use limited labeled anomalies 
in addition with unlabeled data. We address two key limitations in existing work: 
- 1) unlabeled anomalies (i.e., anomaly contamination) may mislead the learning process when all 
     the unlabeled data are employed as inliers for model training; 
- 2) only discrete supervision information (such as binary or ordinal data labels) is exploited, 
     which leads to suboptimal learning of anomaly scores that essentially take on a continuous distribution


Note:

:pushpin:We will include RoSAS into our `DeepOD` python library (https://github.com/xuhongzuo/DeepOD)


## Usage
We provide easy APIs like the sklearn style.
We first instantiate the model class by giving the parameters  
then, the instantiated model can be used to fit and predict data

```python
from RoSAS import RoSAS
model = RoSAS()
model.fit(X_train) # X_train is the training data, use np.array 
score = model.predict(X_test) # X_test is the testing data, use np.array
```


## Citation 

:memo: Please consider citing our paper if you find this repository useful.  

```
@article{xu2022deep,
  title={RoSAS: Deep Semi-supervised Anomaly Detection with Contamination-resilient Continuous Supervision},
  author={Xu, Hongzuo and Wang, Yijie and Pang, Guansong and Jian, Songlei and Wang, Yongjun},
  journal={Information Processing & Management},
  year={2023}
}
```



---