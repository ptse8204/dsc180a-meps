'''
Model_Dev_No_Debias.py contains functions used to render graphs and output of fairness metrics, 
when we train models without de-biasing
'''

#### import packages
import sys
sys.path.insert(0, '../')

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

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
from collections import defaultdict
import matplotlib.pyplot as plt

np.random.seed(1)

### output basic information of the train, valid, and test dataset
def describe(train=None, val=None, test=None):
    if train is not None:
        print("#### Training Dataset shape")
        print(train.features.shape)
    if val is not None:
        print("#### Validation Dataset shape")
        print(val.features.shape)

    if test is not None:
        print("#### Test Dataset shape")
        print(test.features.shape)
        print("#### Favorable and unfavorable labels")
        print(test.favorable_label, test.unfavorable_label)
        print("#### Protected attribute names")
        print(test.protected_attribute_names)
        print("#### Privileged and unprivileged protected attribute values")
        print(test.privileged_protected_attributes, 
              test.unprivileged_protected_attributes)
        print("#### Dataset feature names")
        print(test.feature_names)







def generate_LR_performance_plots_and_charts(train_dataset, valid_dataset, test_dataset):

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': train_dataset.instance_weights}

    lr_orig_panel19 = model.fit(train_dataset.features, train_dataset.labels.ravel(), **fit_params)

    ################################################################################################
    ## Helper Function
    def test(dataset, model, thresh_arr):
        try:
            # sklearn classifier
            y_val_pred_prob = model.predict_proba(dataset.features)
            pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        except AttributeError:
            # aif360 inprocessing algorithm
            y_val_pred_prob = model.predict(dataset).scores
            pos_ind = 0
        
        metric_arrs = defaultdict(list)
        for thresh in thresh_arr:
            y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

            dataset_pred = dataset.copy()
            dataset_pred.labels = y_val_pred
            metric = ClassificationMetric(
                    dataset, dataset_pred,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)

            metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                         + metric.true_negative_rate()) / 2)
            metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
            metric_arrs['disp_imp'].append(metric.disparate_impact())
            metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
            metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
            metric_arrs['theil_ind'].append(metric.theil_index())
        
        return metric_arrs

    ################################################################################################

    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=valid_dataset,
                       model=lr_orig_panel19,
                       thresh_arr=thresh_arr)
    lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])


    ################################################################################################

    def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(x, y_left)
        ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
        ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax1.set_ylim(0.5, 0.8)

        ax2 = ax1.twinx()
        ax2.plot(x, y_right, color='r')
        ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
        if 'DI' in y_right_name:
            ax2.set_ylim(0., 0.7)
        else:
            ax2.set_ylim(-0.25, 0.1)

        best_ind = np.argmax(y_left)
        ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)

    ################################################################################################

    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')


    plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


    ################################################################################################

    def describe_metrics(metrics, thresh_arr):
        best_ind = np.argmax(metrics['bal_acc'])
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
        print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    #     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
        disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
        print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
        print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
        print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
        print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
        print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))



    ###  fairness metrics when testing LR model on original data

    print ("testing LR model on original data")
    lr_orig_metrics = test(dataset=test_dataset,
                       model=lr_orig_panel19,
                       thresh_arr=[thresh_arr[lr_orig_best_ind]])

    describe_metrics(lr_orig_metrics, [thresh_arr[lr_orig_best_ind]])
















def generate_RF_performance_plots_and_charts(train_dataset, valid_dataset, test_dataset):

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'randomforestclassifier__sample_weight': train_dataset.instance_weights}

    lr_orig_panel19 = model.fit(train_dataset.features, train_dataset.labels.ravel(), **fit_params)

    ################################################################################################
    ## Helper Function
    def test(dataset, model, thresh_arr):
        try:
            # sklearn classifier
            y_val_pred_prob = model.predict_proba(dataset.features)
            pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        except AttributeError:
            # aif360 inprocessing algorithm
            y_val_pred_prob = model.predict(dataset).scores
            pos_ind = 0
        
        metric_arrs = defaultdict(list)
        for thresh in thresh_arr:
            y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

            dataset_pred = dataset.copy()
            dataset_pred.labels = y_val_pred
            metric = ClassificationMetric(
                    dataset, dataset_pred,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)

            metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                         + metric.true_negative_rate()) / 2)
            metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
            metric_arrs['disp_imp'].append(metric.disparate_impact())
            metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
            metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
            metric_arrs['theil_ind'].append(metric.theil_index())
        
        return metric_arrs

    ################################################################################################

    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=valid_dataset,
                       model=lr_orig_panel19,
                       thresh_arr=thresh_arr)
    lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])


    ################################################################################################

    def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(x, y_left)
        ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
        ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax1.set_ylim(0.5, 0.8)

        ax2 = ax1.twinx()
        ax2.plot(x, y_right, color='r')
        ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
        if 'DI' in y_right_name:
            ax2.set_ylim(0., 0.7)
        else:
            ax2.set_ylim(-0.25, 0.1)

        best_ind = np.argmax(y_left)
        ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)

    ################################################################################################

    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')


    plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


    ################################################################################################

    def describe_metrics(metrics, thresh_arr):
        best_ind = np.argmax(metrics['bal_acc'])
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
        print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    #     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
        disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
        print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
        print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
        print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
        print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
        print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))



    ###  fairness metrics when testing LR model on original data

    print ("testing RF model on original data")
    lr_orig_metrics = test(dataset=test_dataset,
                       model=lr_orig_panel19,
                       thresh_arr=[thresh_arr[lr_orig_best_ind]])

    describe_metrics(lr_orig_metrics, [thresh_arr[lr_orig_best_ind]])


