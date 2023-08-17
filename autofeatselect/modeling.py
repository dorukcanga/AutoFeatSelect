import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler,FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


def create_preporcessing_pipeline(numeric_columns=None, categorical_columns=None,
                                  num_imputer=None, num_imp_value=None, scaler=None,
                                  cat_imputer=None, cat_imp_value=None, encoder=None):
    
    """
    Create a data preprocessing pipeline using scikit-learn's ColumnTransformer.

    Parameters:
        numeric_columns (list): List of column names corresponding to numeric features.
        categorical_columns (list): List of column names corresponding to categorical features.
        num_imputer (str or None): Strategy for imputing missing values in numeric features.
                                   Options: 'mean', 'median', 'most_frequent', 'knn', None.
        num_imp_value (float or None): Value used for filling missing numeric values when num_imputer is 'constant'.
                                       Ignored for other imputation strategies.
        scaler (str or None): Scaling method for numeric features. Options: 'minmax', 'standard', None.
        cat_imputer (str or None): Strategy for imputing missing values in categorical features.
                                   Options: 'most_frequent', 'constant', 'knn', None.
        cat_imp_value (str, int, or None): Value used for filling missing categorical values when cat_imputer is 'constant' or 'knn'.
                                          Ignored for other imputation strategies.
        encoder (str or None): Encoding method for categorical features. Options: 'ohe', 'te', None.

    Returns:
        preprocessor (ColumnTransformer): A scikit-learn ColumnTransformer object that applies the specified preprocessing steps
                                         to the input data based on the provided configuration.

    Note:
        - The 'num_imputer' parameter can be set to 'knn' to use KNNImputer for imputing numeric features.
        - The 'cat_imputer' parameter can be set to 'knn' to use KNNImputer for imputing categorical features.
        - The 'scaler' parameter can be set to 'minmax' for MinMaxScaler or 'standard' for StandardScaler to scale numeric features.
        - The 'encoder' parameter can be set to 'ohe' for OneHotEncoder or 'te' for TargetEncoder to encode categorical features.
        - If both 'num_imputer' and 'scaler' are set to None, a dummy identity function is applied to numeric features.

    Example:
        preprocessor = create_preprocessing_pipeline(
            numeric_columns=['age', 'income'],
            categorical_columns=['gender', 'education'],
            num_imputer='mean',
            scaler='standard',
            cat_imputer='most_frequent',
            encoder='ohe'
        )
        X_train_preprocessed = preprocessor.fit_transform(X_train)
    """
    
    cat_pipeline_steps, num_pipeline_steps, pre_transformers = [], [], []
    
    #Numeric Pipeline
    if numeric_columns is not None and len(numeric_columns) !=0: 
        
        #Numeric Imputer
        if num_imputer is not None and num_imputer !='knn':
            num_pipeline_steps.append(('imputer', SimpleImputer(strategy=num_imputer, fill_value=num_imp_value)))
        elif num_imputer is not None and num_imputer =='knn':
            num_pipeline_steps.append(('imputer', KNNImputer(n_neighbors=num_imp_value)))
                                      
        #Scaling
        if scaler is not None:
            if scaler == 'minmax':
                num_pipeline_steps.append(('scaler', MinMaxScaler()))
            elif scaler == 'standard':
                num_pipeline_steps.append(('scaler', StandardScaler()))
                                      
        #Identity Function
        if num_imputer is None and scaler is None:
            num_pipeline_steps.append(('identity', FunctionTransformer(func = None)))
                                      
        numeric_transformer = Pipeline(steps=num_pipeline_steps) 
        pre_transformers.append(('numeric_trans', numeric_transformer, numeric_columns))

    #Categorical Pipeline
    if categorical_columns is not None and len(categorical_columns) !=0:

        #Categorical Imputer
        if cat_imputer is not None and cat_imputer !='knn':
            cat_pipeline_steps.append(('imputer', SimpleImputer(strategy=cat_imputer, fill_value=cat_imp_value)))
        elif cat_imputer is not None and cat_imputer =='knn':
            cat_pipeline_steps.append(('imputer', KNNImputer(n_neighbors=cat_imp_value)))
                                      
        #Encoding
        if encoder is not None:
            if encoder == 'ohe':
                cat_pipeline_steps.append(('encoder', OneHotEncoder(sparse = False, handle_unknown = "ignore")))
            elif encoder == 'te':
                cat_pipeline_steps.append(('encoder', TargetEncoder(handle_missing='value', handle_unknown='value')))
                                      
                                      
        categorical_transformer = Pipeline(steps=cat_pipeline_steps)
        pre_transformers.append(('categorical_trans', categorical_transformer, categorical_columns))
                                      
    preprocessor = ColumnTransformer(transformers = pre_transformers, remainder='drop')
    
    return preprocessor

                                      
                                      
def create_modeling_pipeline(preprocessing_pipeline, model_object):
    
    """
    Create a modeling pipeline by combining a data preprocessing pipeline and a machine learning model.

    Parameters:
        preprocessing_pipeline (ColumnTransformer): A scikit-learn ColumnTransformer object that preprocesses the input data.
        model_object: A machine learning model instance.

    Returns:
        modeling_pipeline (Pipeline): A scikit-learn Pipeline object that applies data preprocessing and then fits the model
                                      on the transformed data.

    Example:
        # Assuming 'preprocessor' is a pre-defined ColumnTransformer and 'model' is a pre-defined machine learning model
        modeling_pipeline = create_modeling_pipeline(preprocessor, model)
        modeling_pipeline.fit(X_train, y_train)
        y_pred = modeling_pipeline.predict(X_test)
    """
    
    modeling_pipeline_steps = []
    modeling_pipeline_steps.append(('preprocessor', preprocessing_pipeline))
    modeling_pipeline_steps.append(('model', model_object))
                                      
    modeling_pipeline = Pipeline(steps=modeling_pipeline_steps)
    
    return modeling_pipeline



def create_boosting_model(algorithm, modeling_type, hyperparams, numeric_columns=None, categorical_columns=None, preprocessing_params=None):
    
    """
    Create a boosting model using either LightGBM or XGBoost, with optional data preprocessing.

    Parameters:
        algorithm (str): The algorithm to use for creating the boosting model.
                         Options: 'lightgbm' or 'xgboost'.
        modeling_type (str): The type of modeling task.
                             Options: 'classification' or 'regression'.
        hyperparams (dict): A dictionary of hyperparameters to be passed to the boosting model constructor.
        numeric_columns (list or None): List of column names corresponding to numeric features.
                                        Only needed when preprocessing is applied.
        categorical_columns (list or None): List of column names corresponding to categorical features.
                                            Only needed when preprocessing is applied.
        preprocessing_params (dict or None): A dictionary of preprocessing parameters.
                                             Only needed when preprocessing is applied.
                                             Keys:
                                             - num_imputer: Strategy for imputing missing values in numeric features.
                                               Options: 'mean', 'median', 'most_frequent', 'knn', None.
                                             - num_imp_value: Value used for filling missing numeric values when num_imputer is 'constant'.
                                               Ignored for other imputation strategies.
                                             - scaler: Scaling method for numeric features. Options: 'minmax', 'standard', None.
                                             - cat_imputer: Strategy for imputing missing values in categorical features.
                                               Options: 'most_frequent', 'constant', 'knn', None.
                                             - cat_imp_value: Value used for filling missing categorical values when cat_imputer is 'constant' or 'knn'.
                                               Ignored for other imputation strategies.
                                             - encoder: Encoding method for categorical features. Options: 'ohe', 'te', None.

    Returns:
        model: The boosting model (either LGBMClassifier, LGBMRegressor, XGBClassifier, or XGBRegressor).

    Example:
        # Creating a LightGBM classifier with optional preprocessing
        hyperparams = {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100}
        preprocessing_params = {'num_imputer': 'mean', 'scaler': 'standard', 'encoder': 'ohe'}
        model = create_boosting_model(algorithm='lightgbm', modeling_type='classification',
                                      hyperparams=hyperparams, numeric_columns=['age', 'income'],
                                      categorical_columns=['gender', 'education'], preprocessing_params=preprocessing_params)

        # Creating an XGBoost regressor without preprocessing
        hyperparams = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
        model = create_boosting_model(algorithm='xgboost', modeling_type='regression',
                                      hyperparams=hyperparams)
    """
    
    if algorithm == 'lightgbm':
        if modeling_type == 'classification':
            model = lgb.LGBMClassifier(**hyperparams)
        elif modeling_type == 'regression':
            model = lgb.LGBMRegressor(**hyperparams)
            
    elif algorithm == 'xgboost':
        if modeling_type == 'classification':
            model = xgb.XGBClassifier(**hyperparams)
        elif modeling_type == 'regression':
            model = xgb.XGBRegressor(**hyperparams)
            
    if preprocessing_params is not None:
        preprocessing_pipeline = create_preporcessing_pipeline(numeric_columns=numeric_columns,
                                                               categorical_columns=categorical_columns, **preprocessing_params)
        model = create_modeling_pipeline(preprocessing_pipeline, model)
        return model
    else:
        return model
    
    
