#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(0, '../')

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

from Model_Dev_No_Debias import describe,test,plot,describe_metrics

np.random.seed(1)


# In[ ]:


def training_model(data):
    #dataset = dataset_transf_panel19_train
    model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
    fit_params = {'randomforestclassifier__sample_weight': data.instance_weights}
    rf_transf_panel19 = model.fit(data.features, data.labels.ravel(), **fit_params)
    return rf_transf_panel19


# In[ ]:


def testing_model(val_data,test_data, model, unprivileged_groups, privileged_groups):
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=val_data,
                   model=model,
                   thresh_arr=thresh_arr,
                   unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
    rf_transf_best_ind = np.argmax(val_metrics['bal_acc'])
    
    
    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')
    
    
    plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')
    
    describe_metrics(val_metrics, thresh_arr)
    
    
    rf_transf_metrics = test(dataset=test_data,
                         model=model,
                         thresh_arr=[thresh_arr[rf_transf_best_ind]],
                         unprivileged_groups=unprivileged_groups,
                         privileged_groups=privileged_groups)
    describe_metrics(rf_transf_metrics, [thresh_arr[rf_transf_best_ind]])

