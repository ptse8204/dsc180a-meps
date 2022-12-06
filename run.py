# This file would run the notebook output when given arg == "main"

import sys
import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')
import preprocess
import Model_Dev_No_Debias
import add_model_dev
import reweighing_LR
import reweighing_RF
import Prejudice_Remover

#import eda
#import Model_Dev_No_Debias
#import add_model_dev
#import reweighing_LR
#import reweighing_RF
## Section 6 result summary would be on main
#import explain


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

from collections import OrderedDict
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

from subprocess import getoutput

from collections import defaultdict

import gdown
import zipfile

if __name__ == "__main__":
    all()

def all():
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


    np.random.seed(1)
    
    gdown.download("https://drive.google.com/uc?export=download&id=1YCJVsfkOxrUoaTRLfe59phWBIQO8mG3B", "./", quiet=False)

    with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
        zip_ref.extractall()

    raw_181 = pd.read_csv('h181.csv')

    df_panel_19, df_panel_19_reduced, df_panel_20, df_panel_20_reduced, concat_df = preprocess.get_panel_19_20(raw_181)
    descript_dict, revert_dct = preprocess.description_stats(df_panel_19, df_panel_19_reduced, df_panel_20, df_panel_20_reduced, concat_df)


    os.rename("h181.csv", "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/aif360/data/raw/meps/h181.csv")

    (dataset_orig_panel19_train,
     dataset_orig_panel19_val,
     dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)

    sens_ind = 0
    sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]
                 
    Model_Dev_No_Debias.describe(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test)
    lr_orig_metrics = Model_Dev_No_Debias.generate_model_performance_plots_and_charts(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test, unprivileged_groups, privileged_groups)

    rf_orig_metrics = Model_Dev_No_Debias.generate_RF_performance_plots_and_charts(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test, unprivileged_groups, privileged_groups)

        
    add_model_dev.main(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test, unprivileged_groups, privileged_groups, descript_dict, revert_dct, df_panel_19_reduced)
    
    lr_transf_metrics = reweighing_LR.testing_model(dataset_orig_panel19_val, dataset_orig_panel19_test, reweighing_LR.training_model(reweighing_LR.transform(reweighing_LR.create_dataset(dataset_orig_panel19_train), unprivileged_groups, privileged_groups)), unprivileged_groups, privileged_groups)
    
    rf_transf_metrics = reweighing_RF.testing_model(dataset_orig_panel19_val, dataset_orig_panel19_test, reweighing_RF.training_model(reweighing_LR.transform(reweighing_LR.create_dataset(dataset_orig_panel19_train), unprivileged_groups, privileged_groups)), unprivileged_groups, privileged_groups)
    
    #Prejudice Remover
#    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
#    pr_orig_scaler = StandardScaler()
#
#    dataset = dataset_orig_panel19_train.copy()
#    dataset.features = pr_orig_scaler.fit_transform(dataset.features)
#
#    pr_orig_panel19 = model.fit(dataset)
#
#    thresh_arr = np.linspace(0.01, 0.50, 50)
#
#    dataset = dataset_orig_panel19_val.copy()
#    dataset.features = pr_orig_scaler.transform(dataset.features)
#
#    val_metrics = test(dataset=dataset,
#                       model=pr_orig_panel19,
#                       thresh_arr=thresh_arr)
#
#    pr_orig_best_ind = np.argmax(val_metrics['bal_acc'])
#
#
#    disp_imp = np.array(val_metrics['disp_imp'])
#    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
#    plot(thresh_arr, 'Classification Thresholds',
#         val_metrics['bal_acc'], 'Balanced Accuracy',
#         disp_imp_err, '1 - min(DI, 1/DI)')
#
#    plot(thresh_arr, 'Classification Thresholds',
#    val_metrics['bal_acc'], 'Balanced Accuracy',
#    val_metrics['avg_odds_diff'], 'avg. odds diff.')
#
#    dataset = dataset_orig_panel19_test.copy()
#    dataset.features = pr_orig_scaler.transform(dataset.features)
#
#    pr_orig_metrics = test(dataset=dataset,
#                           model=pr_orig_panel19,
#                           thresh_arr=[thresh_arr[pr_orig_best_ind]])
    
    #Logistic Regression w/ROC
    def compute_metrics(dataset_true, dataset_pred,
                        unprivileged_groups, privileged_groups,
                        disp = True):
        """ Compute the key metrics """
        classified_metric_pred = ClassificationMetric(dataset_true,
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        metrics = OrderedDict()
        metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                                 classified_metric_pred.true_negative_rate())
        metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
        metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
        metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
        metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
        metrics["Theil index"] = classified_metric_pred.theil_index()
        
        if disp:
            for k in metrics:
                print("%s = %.4f" % (k, metrics[k]))
        
        return metrics
        
    dataset = dataset_orig_panel19_train
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(dataset.features)
    y_train = dataset.labels.ravel()

    lr_orig_panel19 = LogisticRegression(solver="liblinear",random_state=1)
    lr_orig_panel19.fit(X_train, y_train)
    y_train_pred = lr_orig_panel19.predict(X_train)

    pos_ind = np.where(lr_orig_panel19.classes_ == dataset.favorable_label)[0][0]
    dataset_orig_train_pred = dataset.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred
    
    dataset_orig_valid_pred = dataset_orig_panel19_val.copy(deepcopy=True)
    X_valid = standard_scaler.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lr_orig_panel19.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_panel19_test.copy(deepcopy=True)
    X_test = standard_scaler.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lr_orig_panel19.predict_proba(X_test)[:,pos_ind].reshape(-1,1)
    
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
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                           +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy:",np.max(ba_arr))
    print("Threshold:", best_class_thresh)
    
    ROC_model = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name="Statistical parity difference",
                                      metric_ub=0.05, metric_lb=-0.05)
    ROC_model = ROC_model.fit(dataset_orig_panel19_val, dataset_orig_valid_pred)
    
    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
    lr_metric_valid_bef = compute_metrics(dataset_orig_panel19_val, dataset_orig_valid_pred, unprivileged_groups, privileged_groups)
    
    #Post-processed validation set
    dataset_post_valid_pred = ROC_model.predict(dataset_orig_valid_pred)
    lr_metric_valid_aft = compute_metrics(dataset_orig_panel19_val, dataset_post_valid_pred, unprivileged_groups, privileged_groups)
    
    #Original test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    lr_metric_test_bef = compute_metrics(dataset_orig_panel19_test, dataset_orig_test_pred, unprivileged_groups, privileged_groups)
    
    #Post-processed test set
    dataset_post_test_pred = ROC_model.predict(dataset_orig_test_pred)
    lr_metric_test_aft = compute_metrics(dataset_orig_panel19_test, dataset_post_test_pred, unprivileged_groups, privileged_groups)
    
    dataset = dataset_orig_panel19_test.copy()
    thresh_arr = np.linspace(0.01, 1.0, 50)
    roc_orig_best_ind = 0
    
    roc_orig_metrics = test(dataset,model=ROC_model,thresh_arr=[thresh_arr[roc_orig_best_ind]])
    
    x = ['Original']
    x2=["Post-processed"]
    #y1 = [1-min(metric_test_bef["Disparate impact"],1/metric_test_bef["Disparate impact"])]
    #y2 = [1-min(metric_test_aft["Disparate impact"],1/metric_test_aft["Disparate impact"])]
    y1=lr_metric_test_bef["Disparate impact"]
    y2=lr_metric_test_aft["Disparate impact"]

    labels=["Before","After"]
    plt.bar(x, y1, color=["grey"],width=0.5)
    plt.bar(x2, y2, color=['b'],width=0.5)
    plt.ylim(0.0, 1.0)
    plt.title("Test: Reject Option Classification Postprocessing")
    plt.ylabel("Disparate Impact")
    plt.legend(labels)
    plt.show()
    #closer to 1 means bias is mitigated
    
    dataset = dataset_orig_panel19_train
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(dataset.features)
    y_train = dataset.labels.ravel()

    rf_orig_panel19 = RandomForestClassifier(n_estimators=500, min_samples_leaf=25,random_state=1)
    rf_orig_panel19.fit(X_train, y_train)
    y_train_pred = rf_orig_panel19.predict(X_train)


    pos_ind = np.where(lr_orig_panel19.classes_ == dataset.favorable_label)[0][0]
    dataset_orig_train_pred = dataset.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred
    
    dataset_orig_valid_pred = dataset_orig_panel19_val.copy(deepcopy=True)
    X_valid = standard_scaler.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = rf_orig_panel19.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_panel19_test.copy(deepcopy=True)
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
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                           +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy:",np.max(ba_arr))
    print("Threshold:", best_class_thresh)
    
    ROC_model = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name="Statistical parity difference",
                                      metric_ub=0.05, metric_lb=-0.05)
    ROC_model = ROC_model.fit(dataset_orig_panel19_val, dataset_orig_valid_pred)
    
    #Original validation set
    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    rf_metric_valid_bef = compute_metrics(dataset_orig_panel19_val, dataset_orig_valid_pred, unprivileged_groups, privileged_groups)
    
    #Post-processed validation set
    dataset_post_valid_pred = ROC_model.predict(dataset_orig_valid_pred)
    rf_metric_valid_aft = compute_metrics(dataset_orig_panel19_val, dataset_post_valid_pred, unprivileged_groups, privileged_groups)
    
    #Original test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    rf_metric_test_bef = compute_metrics(dataset_orig_panel19_test, dataset_orig_test_pred, unprivileged_groups, privileged_groups)
    
    #Post-processed test set
    dataset_post_test_pred = ROC_model.predict(dataset_orig_test_pred)
    rf_metric_test_aft = compute_metrics(dataset_orig_panel19_test, dataset_post_test_pred,unprivileged_groups, privileged_groups)
    
    dataset = dataset_orig_panel19_test.copy()
    thresh_arr = np.linspace(0.01, 1.0, 50)
    roc_orig_best_ind = 0
    
    roc_orig_metrics = test(dataset,
        model=ROC_model,
        thresh_arr=[thresh_arr[roc_orig_best_ind]])
    
    dataset = dataset_orig_panel19_test.copy()
    thresh_arr = np.linspace(0.01, 1.0, 50)
    roc_orig_best_ind = 0
    
    roc_orig_metrics = test(dataset,
        model=ROC_model,
        thresh_arr=[thresh_arr[roc_orig_best_ind]])
    
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
    
    
    # RESULTS
    
    #Results table
    lr_metric_test_df=pd.DataFrame(lr_metric_test_aft,index=[0])[["Balanced accuracy","Average odds difference","Disparate impact","Statistical parity difference","Equal opportunity difference","Theil index"]]
    lr_metric_test_df.columns=["bal_acc","avg_odds_diff","disp_imp","stat_par_diff","eq_opp_diff","theil_ind"]
    rf_metric_test_df=pd.DataFrame(rf_metric_test_aft,index=[0])[["Balanced accuracy","Average odds difference","Disparate impact","Statistical parity difference","Equal opportunity difference","Theil index"]]
    rf_metric_test_df.columns=["bal_acc","avg_odds_diff","disp_imp","stat_par_diff","eq_opp_diff","theil_ind"]
#    list_of_rows=[pd.DataFrame(lr_orig_metrics),pd.DataFrame(rf_orig_metrics),pd.DataFrame(lr_transf_metrics),
#                  pd.DataFrame(rf_transf_metrics),pd.DataFrame(pr_orig_metrics),lr_metric_test_df,rf_metric_test_df]
    list_of_rows=[pd.DataFrame(lr_orig_metrics), pd.DataFrame(rf_orig_metrics), pd.DataFrame(lr_transf_metrics),
                  pd.DataFrame(rf_transf_metrics), lr_metric_test_df, rf_metric_test_df]
    results_table=pd.concat(list_of_rows)
    results_table["Classification"]=["Logistic Regression","Random Forest","Logistic Regression","Random Forest","Logistic Regression"] #"Random Forest"]
    results_table["Stage"]=["Original","Original","Pre-processing","Pre-processing","Post-processing"]#"Post-processing"]
    results_table["Bias Mitigator"]=[" "," ","Reweighing","Reweighing","Reject Option"] #"Reject Option"]
    results_table=results_table.set_index(["Stage","Classification","Bias Mitigator"])
    print(results_table)
    

#def main():
#    eda.main()
#    Model_Dev_No_Debias.main()
#    add_model_dev.main()
#    reweighing_LR.main()
#    reweighing_RF.main()
#    explain.main()
#
##     data_config = json.load(open('config/data-params.json'))
##     eda_config = json.load(open('config/eda-params.json'))
#
##     if 'data' in targets:
#
##         data = generate_data(**data_config)
##         save_data(data, **data_config)
#
##     if 'eda' in targets:
#
##         try:
##             data
##         except NameError:
##             data = pd.read_csv(data_config['data_fp'])
#
##         generate_stats(data, **eda_config)
#
##         # execute notebook / convert to html
##         convert_notebook(**eda_config)
#
#
# if __name__ == '__main__':
#     targets = sys.argv[1:]
#     main()
