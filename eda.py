# Imports
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
import pandas as pd
import seaborn as sns

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from sklearn.pipeline import make_pipeline
import seaborn as sns
from scipy.stats import zscore

def main():
    default_mappings = {
    'label_maps': [{1.0: '>= 10 Visits', 0.0: '< 10 Visits'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-White'}]}

    def default_preprocessing19(df):
        """
        1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
          and 'non-White' otherwise
        2. Restrict to Panel 19
        3. RENAME all columns that are PANEL/ROUND SPECIFIC
        4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
        5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
        """
        def race(row):
            if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE#return 'White'
                return 'Non-White'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df = df.rename(columns = {'RACEV2X' : 'RACE'})
        
        df = df[df['PANEL'] == 19]

        # RENAME COLUMNS
        df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                  'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                  'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                  'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                  'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

        df = df[df['REGION'] >= 0] # remove values -1
        df = df[df['AGE'] >= 0] # remove values -1

        df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP15'] < 10.0
        df.loc[lessE,'TOTEXP15'] = 0.0
        moreE = df['TOTEXP15'] >= 10.0
        df.loc[moreE,'TOTEXP15'] = 1.0

        df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
        return df
    
    def default_preprocessing20(df):
        """
        1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
          and 'non-White' otherwise
        2. Restrict to Panel 20
        3. RENAME all columns that are PANEL/ROUND SPECIFIC
        4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
        5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
        """
        def race(row):
            if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
                return 'White'
            return 'Non-White'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df = df.rename(columns = {'RACEV2X' : 'RACE'})

        df = df[df['PANEL'] == 20]

        # RENAME COLUMNS
        df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                  'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                  'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                  'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                  'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

        df = df[df['REGION'] >= 0] # remove values -1
        df = df[df['AGE'] >= 0] # remove values -1

        df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP15'] < 10.0
        df.loc[lessE,'TOTEXP15'] = 0.0
        moreE = df['TOTEXP15'] >= 10.0
        df.loc[moreE,'TOTEXP15'] = 1.0

        df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
        return df
    
    label_name='UTILIZATION'
    favorable_classes=[1.0]
    protected_attribute_names=['RACE']
    privileged_classes=[['White']]
    instance_weights_name='PERWT15F'
    categorical_features=['REGION','SEX','MARRY',
                                     'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                     'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                     'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                     'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42', 'PHQ242',
                                     'EMPST','POVCAT','INSCOV']

    features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
                                     'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                     'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                     'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                     'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                     'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT15F']
    features_to_drop=[]
    na_values=[]
    # custom_preprocessing=default_preprocessing <- don't need this yet for EDA
    metadata=default_mappings
    
    df_panel_19 = default_preprocessing19(raw_181)
    df_panel_19_reduced = df_panel_19[features_to_keep]
    
    df_panel_20 = default_preprocessing20(raw_181)
    df_panel_20_reduced = df_panel_20[features_to_keep]
    
    ## End of preprocessing
    
    display(df_panel_19_reduced.head(5))
    display(df_panel_20_reduced.head(5))
    features_description = pd.read_html("https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H181")
    # Features description pulled from the website
    description_df = features_description[7]
    display(features_description[7].head())
    descript_dict = dict(zip(description_df.Name.to_list(), description_df.Description.to_list()))
    revert_dct = {'FTSTU': 'FTSTU53X',
     'ACTDTY': 'ACTDTY53',
     'HONRDC': 'HONRDC53',
     'RTHLTH': 'RTHLTH53',
     'MNHLTH': 'MNHLTH53',
     'CHBRON': 'CHBRON53',
     'JTPAIN': 'JTPAIN53',
     'PREGNT': 'PREGNT53',
     'WLKLIM': 'WLKLIM53',
     'ACTLIM': 'ACTLIM53',
     'SOCLIM': 'SOCLIM53',
     'COGLIM': 'COGLIM53',
     'EMPST': 'EMPST53',
     'REGION': 'REGION53',
     'MARRY': 'MARRY53X',
     'AGE': 'AGE53X',
     'POVCAT': 'POVCAT15',
     'INSCOV': 'INSCOV15'}
    print(list(map(descript_dict.get, list(map(lambda x: revert_dct.get(x) if revert_dct.get(x) != None else x, df_panel_19_reduced.columns)))))
    
    des_con = pd.concat([df_panel_19_reduced, df_panel_20_reduced]).describe()
    display(des_con)
    concat_df = pd.concat([df_panel_19_reduced, df_panel_20_reduced])
    display(concat_df["UTILIZATION"].value_counts())
    des_con_1 = concat_df[concat_df["UTILIZATION"] == 1].describe()
    display(des_con_1)
    
    display(des_con_1.loc["mean"] - des_con.loc["mean"] )
    
    null_handled_concat = concat_df
    zscores = concat_df.loc[:, null_handled_concat.columns !=
                                      'RACE'].apply(zscore)
    display(zscores)
    outlier = (zscores < -3) | (zscores > 3)
    display(outlier.sum())
    display(concat_df.isna().sum())
    
    corr_df = concat_df.corr()
    display(corr_df[corr_df != 1].idxmax())
    display(corr_df[corr_df != 1].idxmax().value_counts())
    
    sns.pairplot(concat_df, y_vars =["EMPHDX"])
    sns.pairplot(concat_df, y_vars =["AGE", "PCS42", "MCS42", "PERWT15F"])
    
def test():
    default_mappings = {
    'label_maps': [{1.0: '>= 10 Visits', 0.0: '< 10 Visits'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-White'}]}

    def default_preprocessing19(df):
        """
        1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
          and 'non-White' otherwise
        2. Restrict to Panel 19
        3. RENAME all columns that are PANEL/ROUND SPECIFIC
        4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
        5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
        """
        def race(row):
            if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE#return 'White'
                return 'Non-White'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df = df.rename(columns = {'RACEV2X' : 'RACE'})
        
        df = df[df['PANEL'] == 19]

        # RENAME COLUMNS
        df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                  'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                  'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                  'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                  'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

        df = df[df['REGION'] >= 0] # remove values -1
        df = df[df['AGE'] >= 0] # remove values -1

        df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP15'] < 10.0
        df.loc[lessE,'TOTEXP15'] = 0.0
        moreE = df['TOTEXP15'] >= 10.0
        df.loc[moreE,'TOTEXP15'] = 1.0

        df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
        return df
    
    def default_preprocessing20(df):
        """
        1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
          and 'non-White' otherwise
        2. Restrict to Panel 20
        3. RENAME all columns that are PANEL/ROUND SPECIFIC
        4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
        5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
        """
        def race(row):
            if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
                return 'White'
            return 'Non-White'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df = df.rename(columns = {'RACEV2X' : 'RACE'})

        df = df[df['PANEL'] == 20]

        # RENAME COLUMNS
        df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                  'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                  'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                  'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                  'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

        df = df[df['REGION'] >= 0] # remove values -1
        df = df[df['AGE'] >= 0] # remove values -1

        df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP15'] < 10.0
        df.loc[lessE,'TOTEXP15'] = 0.0
        moreE = df['TOTEXP15'] >= 10.0
        df.loc[moreE,'TOTEXP15'] = 1.0

        df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
        return df
    
    label_name='UTILIZATION'
    favorable_classes=[1.0]
    protected_attribute_names=['RACE']
    privileged_classes=[['White']]
    instance_weights_name='PERWT15F'
    categorical_features=['REGION','SEX','MARRY',
                                     'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                     'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                     'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                     'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42', 'PHQ242',
                                     'EMPST','POVCAT','INSCOV']

    features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
                                     'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                     'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                     'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                     'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                     'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT15F']
    features_to_drop=[]
    na_values=[]
    # custom_preprocessing=default_preprocessing <- don't need this yet for EDA
    metadata=default_mappings
    
    df_panel_19 = default_preprocessing19(raw_181)
    df_panel_19_reduced = df_panel_19[features_to_keep]
    
    df_panel_20 = default_preprocessing20(raw_181)
    df_panel_20_reduced = df_panel_20[features_to_keep]
    
    ## End of preprocessing
    
    display(df_panel_19_reduced.head(5))
    display(df_panel_20_reduced.head(5))
    features_description = pd.read_html("https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H181")
    # Features description pulled from the website
    description_df = features_description[7]
    display(features_description[7].head())
    descript_dict = dict(zip(description_df.Name.to_list(), description_df.Description.to_list()))
    revert_dct = {'FTSTU': 'FTSTU53X',
     'ACTDTY': 'ACTDTY53',
     'HONRDC': 'HONRDC53',
     'RTHLTH': 'RTHLTH53',
     'MNHLTH': 'MNHLTH53',
     'CHBRON': 'CHBRON53',
     'JTPAIN': 'JTPAIN53',
     'PREGNT': 'PREGNT53',
     'WLKLIM': 'WLKLIM53',
     'ACTLIM': 'ACTLIM53',
     'SOCLIM': 'SOCLIM53',
     'COGLIM': 'COGLIM53',
     'EMPST': 'EMPST53',
     'REGION': 'REGION53',
     'MARRY': 'MARRY53X',
     'AGE': 'AGE53X',
     'POVCAT': 'POVCAT15',
     'INSCOV': 'INSCOV15'}
    print(list(map(descript_dict.get, list(map(lambda x: revert_dct.get(x) if revert_dct.get(x) != None else x, df_panel_19_reduced.columns)))))
    
    des_con = pd.concat([df_panel_19_reduced, df_panel_20_reduced]).describe()
    display(des_con)
    concat_df = pd.concat([df_panel_19_reduced, df_panel_20_reduced])
    display(concat_df["UTILIZATION"].value_counts())
    des_con_1 = concat_df[concat_df["UTILIZATION"] == 1].describe()
    display(des_con_1)
    
    display(des_con_1.loc["mean"] - des_con.loc["mean"] )
    
    null_handled_concat = concat_df
    zscores = concat_df.loc[:, null_handled_concat.columns !=
                                      'RACE'].apply(zscore)
    display(zscores)
    outlier = (zscores < -3) | (zscores > 3)
    display(outlier.sum())
    display(concat_df.isna().sum())
    
    corr_df = concat_df.corr()
    display(corr_df[corr_df != 1].idxmax())
    display(corr_df[corr_df != 1].idxmax().value_counts())
    
    sns.pairplot(concat_df, y_vars =["EMPHDX"])
    sns.pairplot(concat_df, y_vars =["AGE", "PCS42", "MCS42", "PERWT15F"])
