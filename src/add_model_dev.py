
#Section 4 Additional Model Development

import sys
sys.path.insert(0, '../')

# %matplotlib inline
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


#From section 3
from Model_Dev_No_Debias import describe,test,plot,describe_metrics
def main(describe,test,plot,describe_metrics):
 np.random.seed(1)

 (dataset_orig_panel19_train,
  dataset_orig_panel19_val,
  dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)

 sens_ind = 0
 sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

 unprivileged_groups = [{sens_attr: v} for v in
                        dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
 privileged_groups = [{sens_attr: v} for v in
                      dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]

 list(map(descript_dict.get, list(map(lambda x: revert_dct.get(x) if revert_dct.get(x) != None else x, df_panel_19_reduced.columns))))

 train_age = pd.DataFrame(dict(zip(["AGE",">=5", "<5", ">17", "<=17", ">23", "<=23"], 
                  [dataset_orig_panel19_train.convert_to_dataframe()[0]["AGE"],
               dataset_orig_panel19_train.convert_to_dataframe()[0]["AGE"] >= 5,
               dataset_orig_panel19_train.convert_to_dataframe()[0]["AGE"] < 5,
               dataset_orig_panel19_train.convert_to_dataframe()[0]["AGE"] > 17,
               dataset_orig_panel19_train.convert_to_dataframe()[0]["AGE"] <= 17,
               dataset_orig_panel19_train.convert_to_dataframe()[0]["AGE"] > 23,
               dataset_orig_panel19_train.convert_to_dataframe()[0]["AGE"] <= 23
               ])))

 val_age = pd.DataFrame(dict(zip(["AGE",">=5", "<5", ">17", "<=17", ">23", "<=23"], 
                  [dataset_orig_panel19_val.convert_to_dataframe()[0]["AGE"],
               dataset_orig_panel19_val.convert_to_dataframe()[0]["AGE"] >= 5,
               dataset_orig_panel19_val.convert_to_dataframe()[0]["AGE"] < 5,
               dataset_orig_panel19_val.convert_to_dataframe()[0]["AGE"] > 17,
               dataset_orig_panel19_val.convert_to_dataframe()[0]["AGE"] <= 17,
               dataset_orig_panel19_val.convert_to_dataframe()[0]["AGE"] > 23,
               dataset_orig_panel19_val.convert_to_dataframe()[0]["AGE"] <= 23
               ])))
 test_age = pd.DataFrame(dict(zip(["AGE",">=5", "<5", ">17", "<=17", ">23", "<=23"], 
                  [dataset_orig_panel19_test.convert_to_dataframe()[0]["AGE"],
               dataset_orig_panel19_test.convert_to_dataframe()[0]["AGE"] >= 5,
               dataset_orig_panel19_test.convert_to_dataframe()[0]["AGE"] < 5,
               dataset_orig_panel19_test.convert_to_dataframe()[0]["AGE"] > 17,
               dataset_orig_panel19_test.convert_to_dataframe()[0]["AGE"] <= 17,
               dataset_orig_panel19_test.convert_to_dataframe()[0]["AGE"] > 23,
               dataset_orig_panel19_test.convert_to_dataframe()[0]["AGE"] <= 23
               ])))

 age_lr_model = LogisticRegression(solver='liblinear', random_state=1)
 age_lr_model.fit(train_age, dataset_orig_panel19_train.labels.ravel())

 # Importing tools for crossValidation
 from sklearn.model_selection import PredefinedSplit
 from sklearn.model_selection import GridSearchCV

 # Create a list where train data indices are -1 and validation data indices are 0
 split_index = ([-1]*(train_age.shape[0])
                   +[0]*(val_age.shape[0]))
 # Use the list to create PredefinedSplit
 pds = PredefinedSplit(test_fold = split_index)

 parameters = {"penalty": ["l1", "l2", "elasticnet", "none"],
               "dual": [True, False],
               "C": 10**np.arange(-4, 4, dtype=float),
               "fit_intercept": [True, False],
               "max_iter": [3000]
 }

 clf_age_lr_model = GridSearchCV(age_lr_model, parameters, cv=pds, scoring='balanced_accuracy').fit(
   pd.concat([train_age, val_age]), 
   np.append(dataset_orig_panel19_train.labels, dataset_orig_panel19_val.labels).ravel())

 clf_age_lr_model.score(val_age, dataset_orig_panel19_val.labels.ravel())

 clf_age_lr_model.score(test_age, dataset_orig_panel19_test.labels.ravel())

 #Feature selection using recursive feature elimination
 from sklearn.feature_selection import RFE
 dataset = dataset_orig_panel19_train
 rfe=RFE(LogisticRegression(solver='liblinear', random_state=1),n_features_to_select=70)
 model = make_pipeline(StandardScaler(),rfe,LogisticRegression(solver='liblinear', random_state=1))
 fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
 lr_new_panel19=model.fit(dataset.features, dataset.labels.ravel())

 thresh_arr = np.linspace(0.01, 0.5, 50)
 val_metrics = test(dataset=dataset_orig_panel19_val,
                    model=lr_new_panel19,
                    thresh_arr=thresh_arr)
 lr_new_best_ind = np.argmax(val_metrics['bal_acc'])

 disp_imp = np.array(val_metrics['disp_imp'])
 disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
 plot(thresh_arr, 'Classification Thresholds',
      val_metrics['bal_acc'], 'Balanced Accuracy',
      disp_imp_err, '1 - min(DI, 1/DI)')

 plot(thresh_arr, 'Classification Thresholds',
      val_metrics['bal_acc'], 'Balanced Accuracy',
      val_metrics['avg_odds_diff'], 'avg. odds diff.')

 describe_metrics(val_metrics, thresh_arr)

 lr_new_metrics = test(dataset=dataset_orig_panel19_test,
                        model=lr_new_panel19,
                        thresh_arr=[thresh_arr[lr_new_best_ind]])

 describe_metrics(lr_new_metrics, [thresh_arr[lr_new_best_ind]])

 #Feature selection using recursive feature elimination
 dataset = dataset_orig_panel19_train
 rfe=RFE(RandomForestClassifier(n_estimators=500, min_samples_leaf=25),n_features_to_select=70)
 model = make_pipeline(StandardScaler(),rfe,RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
 fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
 rf_new_panel19=model.fit(dataset.features, dataset.labels.ravel())

 thresh_arr = np.linspace(0.01, 0.5, 50)
 val_metrics = test(dataset=dataset_orig_panel19_val,
                    model=rf_new_panel19,
                    thresh_arr=thresh_arr)
 rf_new_best_ind = np.argmax(val_metrics['bal_acc'])

 disp_imp = np.array(val_metrics['disp_imp'])
 disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
 plot(thresh_arr, 'Classification Thresholds',
      val_metrics['bal_acc'], 'Balanced Accuracy',
      disp_imp_err, '1 - min(DI, 1/DI)')

 plot(thresh_arr, 'Classification Thresholds',
      val_metrics['bal_acc'], 'Balanced Accuracy',
      val_metrics['avg_odds_diff'], 'avg. odds diff.')

 describe_metrics(val_metrics, thresh_arr)

 rf_new_metrics = test(dataset=dataset_orig_panel19_test,
                        model=rf_new_panel19,
                        thresh_arr=[thresh_arr[rf_new_best_ind]])

 describe_metrics(rf_new_metrics, [thresh_arr[rf_new_best_ind]])
