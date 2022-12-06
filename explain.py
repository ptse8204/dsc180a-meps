from aif360.datasets.lime_encoder import LimeEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer
# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from collections import defaultdict

def main():
    (dataset_orig_panel19_train,
     dataset_orig_panel19_val,
     dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)
    
    sens_ind = 0
    sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]
     
    unprivileged_groups = [{sens_attr: v} for v in
                       dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                     dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)

    lime_data = LimeEncoder().fit(dataset_transf_panel19_train)
    
    #  The transform() method is then used to convert aif360 features to LIME-compatible features
    s_train = lime_data.transform(dataset_transf_panel19_train.features)
    # Any set would work, given u want to use it for prediction
    s_test = lime_data.transform(dataset_orig_panel19_test.features) 
    
    def lr_model_gen(dataset_transf_panel19_train_gen):
        dataset = dataset_transf_panel19_train_gen
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(solver='liblinear', random_state=1))
        fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
        lr_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
        return lr_transf_panel19
    
    
    #model used below:
    model = lr_model_gen(dataset_transf_panel19_train)  # lr_transf_panel19 is LR model learned from Panel 19 with Reweighing

    #The LimeTabularExplainer takes as input the LIME-compatible data along with various other arguments to create a lime explainer
    explainer = LimeTabularExplainer(
        s_train, class_names=lime_data.s_class_names, 
        feature_names=lime_data.s_feature_names,
        categorical_features=lime_data.s_categorical_features, 
        categorical_names=lime_data.s_categorical_names, 
        kernel_width=3, verbose=False, discretize_continuous=True)
    
    # Find which index has different value, randomly picked from the dataset
    mod_pred = model.predict(dataset_orig_panel19_test.features)
    diff_index = np.where((dataset_orig_panel19_test.labels).flatten() != mod_pred)[0]
    wrong_pre_i = np.random.choice(diff_index)
    correct_i = diff_index[0]
    while correct_i in diff_index:
      correct_i = np.random.choice(np.arange(0, len(mod_pred)))
    
    def s_predict_fn(x):
      return model.predict_proba(lime_data.inverse_transform(x))
    
    def show_explanation(ind):
      exp = explainer.explain_instance(s_test[ind], s_predict_fn, num_features=10)
      print("Actual label: " + str(dataset_orig_panel19_test.labels[ind]))
      exp.as_pyplot_figure()
      exp.show_in_notebook()
      plt.show()
    
    print("Predicted correctly by the model")
    show_explanation(correct_i)
    print("Predicted incorrectly by the model")
    show_explanation(wrong_pre_i)

    
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

    def describe_metrics(metrics, thresh_arr):
      best_ind = np.argmax(metrics['bal_acc'])
      print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
      print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
      disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
      print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
      print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
      print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
      print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
      print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
    thresh_arr = np.linspace(0.01, 0.5, 50)
    lr_transf_metrics = test(dataset=dataset_orig_panel19_test,
                            model=lr_model_gen(dataset_transf_panel19_train),
                            thresh_arr=thresh_arr)
    lr_transf_best_ind = np.argmax(lr_transf_metrics['bal_acc'])

    describe_metrics(lr_transf_metrics, thresh_arr)

    scaler = StandardScaler()
    data_scaled_feat = scaler.fit_transform(dataset_transf_panel19_train.features)
    model_nopip = RandomForestClassifier(n_estimators=500, min_samples_leaf=25)
    rf_transf_panel19_nopipe = model_nopip.fit(data_scaled_feat, dataset_transf_panel19_train.labels.ravel(), sample_weight=dataset_transf_panel19_train.instance_weights)

    # Find which index has different value, randomly picked from the dataset
    mod_pred = rf_transf_panel19_nopipe.predict(dataset_orig_panel19_test.features)
    diff_index = np.where((dataset_orig_panel19_test.labels).flatten() != mod_pred)[0]
    wrong_pre_i = np.random.choice(diff_index)
    correct_i = diff_index[0]
    while correct_i in diff_index:
      correct_i = np.random.choice(np.arange(0, len(mod_pred)))
    
    print("Correct Classification:")
    print("The correct labeling is: " + str((dataset_orig_panel19_test.labels).flatten()[correct_i]))

    plt.bar(["< 10 Visits",">= 10 Visits"],rf_transf_panel19_nopipe.predict_proba([dataset_orig_panel19_test.features[correct_i]])[0])
    plt.show()

    print("Incorrect Classification:")
    print("The correct labeling is: " + str((dataset_orig_panel19_test.labels).flatten()[wrong_pre_i]))

    plt.bar(["< 10 Visits",">= 10 Visits"],rf_transf_panel19_nopipe.predict_proba([dataset_orig_panel19_test.features[wrong_pre_i]])[0])
    plt.show()
