# LIME
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

def explain_notebook():
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
