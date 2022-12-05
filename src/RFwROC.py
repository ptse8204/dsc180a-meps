#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import OrderedDict 

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


from aif360.algorithms.postprocessing.reject_option_classification        import RejectOptionClassification


np.random.seed(1)


# In[ ]:


def model(data_train, data_valid, data_test):
    
    #dataset = dataset_orig_panel19_train
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(data_train.features)
    y_train = data_train.labels.ravel()

    rf_orig_panel19 = RandomForestClassifier(n_estimators=500, min_samples_leaf=25,random_state=1)
    rf_orig_panel19.fit(X_train, y_train)
    y_train_pred = rf_orig_panel19.predict(X_train)


    pos_ind = np.where(lr_orig_panel19.classes_ == data_train.favorable_label)[0][0]
    dataset_orig_train_pred = data_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred
    
    #validate model
    dataset_orig_valid_pred = data_valid.copy(deepcopy=True)
    X_valid = standard_scaler.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = rf_orig_panel19.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    #test model
    dataset_orig_test_pred = data_test.copy(deepcopy=True)
    X_test = standard_scaler.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = rf_orig_panel19.predict_proba(X_test)[:,pos_ind].reshape(-1,1)
    
    
    num_thresh = 50
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.5, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_panel19_val,
                                             dataset_orig_valid_pred, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()                       +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy:",np.max(ba_arr))
    print("Threshold:", best_class_thresh)
    
    
    
    
    #ROC Model 
    ROC_model = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name="Statistical parity difference",
                                  metric_ub=0.05, metric_lb=-0.05)
    ROC_model = ROC_model.fit(dataset_orig_panel19_val, dataset_orig_valid_pred)
    
    
    #evalute on validation set 
    #Original validation set
    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    rf_metric_valid_bef = compute_metrics(dataset_orig_panel19_val, dataset_orig_valid_pred, unprivileged_groups, privileged_groups)
    
    
    #Post-processed validation set
    dataset_post_valid_pred = ROC_model.predict(dataset_orig_valid_pred)
    rf_metric_valid_aft = compute_metrics(dataset_orig_panel19_val, dataset_post_valid_pred, unprivileged_groups, privileged_groups)
    
    
    #evaluate on test set
    #Original test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    rf_metric_test_bef = compute_metrics(dataset_orig_panel19_test, dataset_orig_test_pred, unprivileged_groups, privileged_groups)
    
    
    #Post-processed test set
    dataset_post_test_pred = ROC_model.predict(dataset_orig_test_pred)
    rf_metric_test_aft = compute_metrics(dataset_orig_panel19_test, dataset_post_test_pred,unprivileged_groups, privileged_groups)
    
    
    
    
    thresh_arr = np.linspace(0.01, 1.0, 50)
    roc_orig_best_ind = np.argmax(val_metrics['bal_acc'])
    
    roc_orig_metrics = test(data_test,model=ROC_model,thresh_arr=[thresh_arr[roc_orig_best_ind]])
    
    
    #generate plots 
    x = ['Original']
    x2=["Post-processed"]
    #y1 = [1-min(metric_test_bef["Disparate impact"],1/metric_test_bef["Disparate impact"])]
    #y2 = [1-min(metric_test_aft["Disparate impact"],1/metric_test_aft["Disparate impact"])]
    y1=rf_metric_test_bef["Disparate impact"]
    y2=rf_metric_test_aft["Disparate impact"]

    labels=["Before","After"]
    plt.bar(x, y1, color=["grey"],width=0.5)
    plt.bar(x2, y2, color=['b'],width=0.5)
    plt.ylim(0.0, 1.0)
    plt.title("Test: Reject Option Classification Postprocessing")
    plt.ylabel("Disparate Impact Error")
    plt.legend(labels)
    plt.show()
    #closer to 1 means bias mitigated
    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




