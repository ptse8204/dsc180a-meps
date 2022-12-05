#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(0, '../')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display


# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# LIME
from aif360.datasets.lime_encoder import LimeEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer


#imports from other py
#import test from  
#import describe_metrics from 

np.random.seed(1)


# In[ ]:


#obtain data from EDA 


# In[ ]:


def create_dataset():
    data = dataset_orig_panel19_train.copy()
    
    pr_orig_scaler = StandardScaler()
    data.features = pr_orig_scaler.fit_transform(data.features)
    
    return data


# In[ ]:


def transform(data):
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    dataset_transf_panel19_train = RW.fit_transform(data)
    return dataset_transf_panel19_train


# In[ ]:


def generate_metrics(data):
    metric_transf_panel19_train = BinaryLabelDatasetMetric(
        dataset_transf_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    explainer_transf_panel19_train = MetricTextExplainer(data)

    print(explainer_transf_panel19_train.disparate_impact())
    


# In[ ]:


def training_model(data):
    #dataset = dataset_transf_panel19_train
    model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': data.instance_weights}
    lr_transf_panel19 = model.fit(data.features, data.labels.ravel(), **fit_params)
    
    return lr_transf_panel19 


# In[ ]:


def testing_model(valid_data,test_data, model):
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=valid_data,
                   model=model,
                   thresh_arr=thresh_arr)
    lr_transf_best_ind = np.argmax(val_metrics['bal_acc'])
    
    
    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')
    
    
    plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')
    
    
    describe_metrics(val_metrics, thresh_arr)
    
    
    lr_transf_metrics = test(dataset=test_data,
                         model=model,
                         thresh_arr=[thresh_arr[lr_transf_best_ind]])
    
    describe_metrics(lr_transf_metrics, [thresh_arr[lr_transf_best_ind]])
    

