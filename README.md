# Automated Feature Selection & Importance

`autofeatselect` is a python library that automates and accelerates feature selection processes for machine learning projects.

It helps to calculate feature importance scores & rankings with several methods and also helps to detect and remove highly correlated variables.

## Installation

You can install from PyPI:

```shell
pip install autofeatselect
```

## Key Features

`autofeatselect` offers a wide range of features to support feature selection and importance analysis:

* Automated Feature Selection: Various automated feature selection methods, such as LGBM Importance, XGBoost Importance, RFECV so on. 
* Feature Importance Analysis: Calculation and visualization of feature importance scores for different algorithms seperately.
* Correlation Analysis: Perform correlation analysis to identify and drop correlated features automatically.

## Full List of Methods

__Correlation Calculation Methods__
* Pearson, Spearman & Kendall Correlation Coefficients for Continuous Variables
* Cramer's V Scores for Categorical Variables

__Feature Selection Methods__
* LightGBM Feature Importance Scores
* XGBoost Feature Importance Scores (with Target Encoding for Categorical Variables)
* Random Forest Feature Importance Scores (with Target Encoding for Categorical Variables)
* LassoCV Coefficients (with One Hot Encoding for Categorical Variables)
* Permutation Importance Scores (LightGBM as the estimator)
* RFECV Rankings (LightGBM as the estimator)
* Boruta Rankings (Random Forest as the estimator)

## Usage


* Calculating Correlations & Detecting Highly Correlated Features

```python
num_static_feats = ['x1', 'x2'] #Static features to be kept regardless of the correlation results.

corr_df_num, remove_list_num = CorrelationCalculator.numeric_correlations(X=X_train,
                                                                          features=num_feats, #List of continuous features
                                                                          static_features=num_static_feats,
                                                                          corr_method='pearson',
                                                                          threshold=0.9)

corr_df_cat, remove_list_cat = CorrelationCalculator.categorical_correlations(X=X_train,
                                                                              features=cat_feats, #List of categorical features
                                                                              static_features=None,
                                                                              threshold=0.9)
```

* Calculating Single Feature Importance Score & Plot Results

```python
#Create Feature Selection Object
feat_selector = FeatureSelector(modeling_type='classification', # 'classification' or 'regression'
                                X_train=X_train,
                                y_train=y_train,
                                X_test=None,
                                y_test=None,
                                numeric_columns=num_feats,
                                categorical_columns=cat_feats,
                                seed=24)

#Train LightGBM model & return importance results as pd.DataFrame 
lgbm_importance_df = feat_selector.lgbm_importance(hyperparam_dict=None,
                                                   objective=None,
                                                   return_plot=True)


#Apply RFECV with using LightGBM as the estimator & return importance results as pd.DataFrame 
lgbm_hyperparams = {'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 400,
                    'num_leaves': 30, 'random_state':24, 'importance_type':'gain'
                   }
rfecv_hyperparams = {'step':3, 'min_features_to_select':5, 'cv':5}

rfecv_importance_df = feat_selector.rfecv_importance(lgbm_hyperparams=lgbm_hyperparams,
                                                     rfecv_hyperparams=rfecv_hyperparams,
                                                     return_plot=False)
```

* Calculating Single Feature Importance Score & Plot Results

```python
#Automated correlation analysis & applying multiple feature selection methods
feat_selector = AutoFeatureSelect(modeling_type='classification',
                                  X_train=X_train,
                                  y_train=y_train,
                                  X_test=X_test,
                                  y_test=y_test,
                                  numeric_columns=num_feats,
                                  categorical_columns=cat_feats,
                                  seed=24)

corr_features = feat_selector.calculate_correlated_features(static_features=None,
                                                            num_threshold=0.9,
                                                            cat_threshold=0.9)

feat_selector.drop_correlated_features()

final_importance_df = feat_selector.apply_feature_selection(selection_methods=['lgbm', 'xgb', 'perimp', 'rfecv', 'boruta'])
```

## License

This project is completely free, open-source and licensed under the MIT license.
