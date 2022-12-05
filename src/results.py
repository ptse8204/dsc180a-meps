import pandas as pd

def return_table(lr_metric_test_aft, rf_metric_test_aft, lr_orig_metrics, rf_orig_metrics, lr_transf_metrics, rf_transf_metrics, pr_orig_metrics)):
    lr_metric_test_df=pd.DataFrame(lr_metric_test_aft,index=[0])[["Balanced accuracy","Average odds difference","Disparate impact","Statistical parity difference","Equal opportunity difference","Theil index"]]
    lr_metric_test_df.columns=["bal_acc","avg_odds_diff","disp_imp","stat_par_diff","eq_opp_diff","theil_ind"]
    rf_metric_test_df=pd.DataFrame(rf_metric_test_aft,index=[0])[["Balanced accuracy","Average odds difference","Disparate impact","Statistical parity difference","Equal opportunity difference","Theil index"]]
    rf_metric_test_df.columns=["bal_acc","avg_odds_diff","disp_imp","stat_par_diff","eq_opp_diff","theil_ind"]
    list_of_rows=[pd.DataFrame(lr_orig_metrics),pd.DataFrame(rf_orig_metrics),pd.DataFrame(lr_transf_metrics),
                  pd.DataFrame(rf_transf_metrics),pd.DataFrame(pr_orig_metrics),lr_metric_test_df,rf_metric_test_df]
    results_table=pd.concat(list_of_rows)
    results_table["Classification"]=["Logistic Regression","Random Forest","Logistic Regression","Random Forest",
                                    " ","Logistic Regression","Random Forest"]
    results_table["Stage"]=["Original","Original","Pre-processing","Pre-processing","In-processing","Post-processing","Post-processing"]
    results_table["Bias Mitigator"]=[" "," ","Reweighing","Reweighing","Prejudice Remover","Reject Option","Reject Option"]
    results_table=results_table.set_index(["Stage","Classification","Bias Mitigator"])
    results_table
    
