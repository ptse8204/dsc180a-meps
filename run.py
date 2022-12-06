# This file would run the notebook output when given arg == "main"

import sys
import os
import json

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

from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

import gdown
import zipfile

def all():
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
    
    data, pr_orig_scalar = Prejudice_Remover.create_dataset(dataset_orig_panel19_train)
    
    pr_orig_metrics = Prejudice_Remover.testing_model(dataset_orig_panel19_val, dataset_orig_panel19_test, Prejudice_Remover.create_PR_model(data, sens_attr), pr_orig_scalar, unprivileged_groups, privileged_groups)
    
    
    
    
    
    
def test():
    print("test")


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
