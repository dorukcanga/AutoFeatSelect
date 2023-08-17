import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from category_encoders.target_encoder import TargetEncoder
from sklearn.feature_selection import RFECV

import scipy.stats as ss
from scipy.stats import pearsonr
from boruta import BorutaPy
from functools import reduce

sns.set_style(style='darkgrid')

import lightgbm as lgb
import xgboost as xgb

from .modeling import create_preporcessing_pipeline, create_modeling_pipeline, create_boosting_model



class UtilityFunctions:
    """
    A collection of utility functions for visualizing and plotting feature importance.

    Methods:
    --------
    plot_importance(df: pandas.DataFrame)
        Plots feature importance scores.

    """
    
    @staticmethod
    def plot_importance(df):
        """
        Plots feature importance scores.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing feature names and their importance scores.

        Returns:
        --------
        None
        """
        plt.figure(figsize=(20, 10))
        sns.barplot(x="importance", y="feature", data=df)
        plt.tight_layout()
        plt.show()

class FeatureSelectorConfig:
    """
    A class for storing hyperparameters and configurations for feature selection methods.
    
    Parameters:
    -----------
    seed : int or None, optional
        Seed value for reproducibility.

    Attributes:
    -----------
    seed : int or None, optional
        Seed value for reproducibility.

    lgbm_hyperparams : dict
        Hyperparameters for LightGBM model.

    xgb_hyperparams : dict
        Hyperparameters for XGBoost model.

    rf_hyperparams : dict
        Hyperparameters for RandomForest model.

    lasso_hyperparams : dict
        Hyperparameters for LassoCV model.

    perimp_hyperparams : dict
        Hyperparameters for permutation importance calculation.

    rfecv_hyperparams : dict
        Hyperparameters for Recursive Feature Elimination with Cross-Validation.

    boruta_hyperparams : dict
        Hyperparameters for Boruta feature selection.

    """
    
    def __init__(self, seed=None):

        self.seed = seed
        self.lgbm_hyperparams = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'num_leaves': 20,
                            'random_state':self.seed, 'n_jobs':-1, 'importance_type':'gain', 'verbose':-1
                           }
        
        self.xgb_hyperparams = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200,
                           'random_state':self.seed, 'n_jobs':-1, 'importance_type':'total_gain', 'verbosity':0
                          }
        
        self.rf_hyperparams = {'max_depth': 5, 'n_estimators': 200, 'random_state':self.seed, 'n_jobs':-1, 'verbose':0}
        
        self.lasso_hyperparams = {'cv':5, 'random_state':self.seed, 'n_jobs':-1}

        self.perimp_hyperparams = {'n_repeats':5, 'n_jobs':-1, 'random_state':self.seed}
        
        self.rfecv_hyperparams = {'step':1, 'min_features_to_select':1, 'cv':5}
        
        self.boruta_hyperparams = {'n_estimators':'auto', 'max_iter':100, 'random_state':self.seed}

class CorrelationCalculator:
    """
    Class for calculating feature correlations.
    """
    
    @staticmethod
    def numeric_correlations(X, features, static_features=None, corr_method='pearson', threshold=0.9):
        """
        Calculate numeric feature correlations with pearson, spearman or kendall correlation coefficient.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input DataFrame containing the features.

        features : List[str]
            List of feature names for which correlations will be calculated.

        static_features : List[str] (default=None)
            List of feature names that are considered static and should not be dropped due to correlations.

        corr_method : str (default='pearson')
            Method to compute correlations (e.g., 'pearson', 'spearman', 'kendall').

        threshold : float (default=0.9)
            Correlation threshold above which features are considered correlated and may be removed.

        Returns:
        --------
        corr_df : pandas.DataFrame
            DataFrame containing correlation scores between features.

        remove_list : List[str]
            List of features to be removed due to high correlation with other features.
            
        Examples:
        ---------
        numeric_corr, corr_features = calculator.numeric_correlations(X=X_train, features=num_features, static_features=static_features)

        """
        
        if static_features is None:
            static_features = []
        
        # Compute the correlation matrix using Pearson correlation
        corr_matrix = X[features].corr(method=corr_method).abs()
        
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1)
        corr_matrix = corr_matrix.where(upper_triangle == 0)
        
        # Melt the DataFrame to convert it to long format
        corr_matrix = corr_matrix.reset_index()
        melted_df = pd.melt(corr_matrix, id_vars='index', value_vars=corr_matrix.columns[1:], var_name='j')
        
        # Rename columns for clarity
        melted_df.columns = ['i', 'j', 'correlation_score']
        
        # Drop the rows where 'Feature 1' is equal to 'Feature 2'
        melted_df = melted_df[melted_df['i'] != melted_df['j']]
        melted_df = melted_df[melted_df.correlation_score.isna()==False].sort_values('correlation_score', ascending=False)
        
        # Find and remove one of the correlated features
        remove_list = []
        filtered_df = melted_df[melted_df.correlation_score > threshold]
        
        check = True if len(filtered_df) > 0 else False
        while check:
            feature_i, feature_j = filtered_df.i.values[0], filtered_df.j.values[0]

            feature_to_keep = feature_j if feature_j in static_features else feature_i
            removed_feature = feature_i if feature_to_keep == feature_j else feature_j
            removed_feature = None if removed_feature in static_features else removed_feature

            if removed_feature is not None:
                remove_list.append(removed_feature)
                filtered_df = filtered_df[(filtered_df.i != removed_feature) & (filtered_df.j != removed_feature)]
            else:
                filtered_df = filtered_df.iloc[1:,]

            df_len = len(filtered_df)
            remaining_feats = list(set(list(filtered_df.i.values) + list(filtered_df.j.values)))
            check_list = list(set(remaining_feats) - set(static_features))

            if df_len == 0 or len(check_list) == 0:
                check = False
                
        return melted_df, remove_list
    
    
    @staticmethod
    def categorical_correlations(X, features, static_features=None, threshold=0.9):
        """
        Calculate categorical feature correlations with Cramer's V.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input DataFrame containing the features.

        features : List[str]
            List of feature names for which correlations will be calculated.

        static_features : List[str] (default=None)
            List of feature names that are considered static and should not be dropped due to correlations.

        threshold : float (default=0.9)
            Correlation threshold above which features are considered correlated and may be removed.

        Returns:
        --------
        corr_df : pandas.DataFrame
            DataFrame containing correlation scores between features.

        remove_list : List[str]
            List of features to be removed due to high correlation with other features.
            
        Examples:
        ---------
        cat_corr, corr_features = calculator.categorical_correlations(X=X_train, features=cat_features, static_features=static_features)
        """
        
        if static_features is None:
            static_features = []
        
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x,y)
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rcorr = r-((r-1)**2)/(n-1)
            kcorr = k-((k-1)**2)/(n-1)
            return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
        
        # Compute the correlation matrix using Cramers V Correlation
        categorical_columns = [i for i in features if X[i].nunique() != 1]
        num_features = len(categorical_columns)  
        unique_category_counts = {col: len(X[col].unique()) for col in categorical_columns}

        cramers_v_matrix = np.zeros((len(categorical_columns), len(categorical_columns)))
        
        for i, col_i in enumerate(categorical_columns):
            for j, col_j in enumerate(categorical_columns):
                cramers_v_matrix[i, j] = cramers_v(X[col_i], X[col_j])
                
        cramers_v_df = pd.DataFrame(cramers_v_matrix, index=categorical_columns, columns=categorical_columns)
        
        upper_triangle = np.triu(np.ones(cramers_v_df.shape), k=1)
        cramers_v_df = cramers_v_df.where(upper_triangle == 0)
        np.fill_diagonal(cramers_v_df.values, np.nan)

        # Melt the DataFrame to convert it to long format
        melted_df = cramers_v_df.stack().reset_index()
        melted_df.columns = ['i', 'j', 'correlation_score']

        remove_list = []
        filtered_df = melted_df[melted_df.correlation_score > threshold]
        
        # Find and remove one of the correlated features
        check = True if len(filtered_df) > 0 else False
        while check:
            feature_i, feature_j = filtered_df.i.values[0], filtered_df.j.values[0]

            feature_to_keep = feature_i if unique_category_counts[feature_i] >= unique_category_counts[feature_j] else feature_j
            removed_feature = feature_i if feature_to_keep == feature_j else feature_j
            removed_feature = None if removed_feature in static_features else removed_feature

            if removed_feature is not None:
                remove_list.append(removed_feature)
                filtered_df = filtered_df[(filtered_df.i != removed_feature) & (filtered_df.j != removed_feature)]
            else:
                filtered_df = filtered_df.iloc[1:,]

            df_len = len(filtered_df)
            remaining_feats = list(set(list(filtered_df.i.values) + list(filtered_df.j.values)))
            check_list = list(set(remaining_feats) - set(static_features))

            if df_len == 0 or len(check_list) == 0:
                check = False
                
        return melted_df, remove_list

class FeatureSelector:
    
    def __init__(self, modeling_type, X_train, y_train, weight_train=None,
                 X_test=None, y_test=None,
                 numeric_columns=None, categorical_columns=None,
                 seed=None):
        """
        Class for selecting and evaluating features using different importance methods.

        Parameters:
        -----------
        modeling_type : str
            Type of modeling task ('regression' or 'classification').

        X_train : pandas.DataFrame
            Training data

        y_train : pandas.Series
            Training labels

        weight_train : pandas.Series, optional (default=None)
            Weights for the training data.

        X_test : pandas.DataFrame, optional (default=None)
            Test data

        y_test : pandas.Series, optional (default=None)
            Test labels

        numeric_columns : List[str], optional (default=None)
            List of column names containing numeric features.

        categorical_columns : List[str], optional (default=None)
            List of column names containing categorical features.

        seed : int or None, optional (default=None)
            Seed value for reproducibility.
            
        Methods:
        --------
        lgbm_importance(hyperparam_dict: dict, objective: str, return_plot: bool)
            Calculate feature importance using LightGBM.

        xgb_importance(hyperparam_dict: dict, objective: str, return_plot: bool)
            Calculate feature importance using XGBoost.

        rf_importance(hyperparam_dict: dict, return_plot: bool)
            Calculate feature importance using RandomForest.

        lassocv_coefs(hyperparam_dict: dict, return_plot: bool)
            Calculate LassoCV feature coefficients.

        permutation_importance(lgbm_hyperparams: dict, perimp_hyperparams: dict, return_plot: bool)
            Calculate feature importance using permutation importance.

        rfecv_importance(lgbm_hyperparams: dict, rfecv_hyperparams: dict, return_plot: bool)
            Calculate feature importance using Recursive Feature Elimination with Cross-Validation.

        boruta_rankings(rf_hyperparams: dict, boruta_hyperparams: dict)
            Calculate feature rankings using Boruta feature selection.
            
        Returns:
        --------
        The respective feature importance DataFrames or rankings obtained from each method.
        
        Examples:
        ---------
        # Example usage:
        feat_selector = FeatureSelector(modeling_type='regression', X_train=X_train, y_train=y_train,
                                   numeric_columns=num_features, categorical_columns=cat_features, seed=42)
                                   
        lgbm_importance_df = feat_selector.lgbm_importance(return_plot=True)

        perimp_importance_df = feat_selector.permutation_importance(lgbm_hyperparams=lgbm_hyperparams, perimp_hyperparams=perimp_hyperparams)
        """
        
        self.modeling_type = modeling_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.weight_train = weight_train
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.seed = seed
        
        #Data Type Correction for Training & Test Sets
        col_list = []
        if self.numeric_columns is not None:
            col_list = col_list + self.numeric_columns
        if self.categorical_columns is not None:
            col_list = col_list + self.categorical_columns
        self.X_train = self.X_train[col_list]
        
        self.X_train.loc[len(self.X_train)] = [np.nan] * len(self.numeric_columns) + ['unknown'] * len(self.categorical_columns)
        self.y_train.loc[len(self.y_train)] = self.y_train.mode().values[0]
        
        if self.numeric_columns is not None:
            self.X_train[self.numeric_columns] = self.X_train[self.numeric_columns].astype('float')
        if self.categorical_columns is not None:
            self.X_train[self.categorical_columns] = self.X_train[self.categorical_columns].astype('category')
        
        if self.X_test is not None:
            self.X_test = self.X_test[col_list]
            
            if self.numeric_columns is not None:
                self.X_test[self.numeric_columns] = self.X_test[self.numeric_columns].astype('float')
            if self.categorical_columns is not None:
                self.X_test[self.categorical_columns] = self.X_test[self.categorical_columns].astype('category')
            
            
    def lgbm_importance(self, hyperparam_dict=None, objective=None, return_plot=False):
        """
        Calculate feature importance using LightGBM model.

        Parameters:
            hyperparam_dict (dict): Hyperparameters for the LightGBM model. (Optional)
            objective (str): Objective function for the LightGBM model. (Optional)
            return_plot (bool): Whether to return an importance plot.

        Returns:
            lgbm_importance_df (pd.DataFrame): DataFrame with feature importances.
        """


        if hyperparam_dict == None:
            hyperparam_dict = FeatureSelectorConfig(seed=self.seed).lgbm_hyperparams
        if objective is not None:
            hyperparam_dict['objective'] = objective

        #Model Training
        model = create_boosting_model(algorithm='lightgbm',
                                      modeling_type=self.modeling_type,
                                      hyperparams=hyperparam_dict)

        model.fit(X=self.X_train, y=self.y_train, sample_weight=self.weight_train)

        #Feature Importance DataFrame
        self.lgbm_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'importance' : model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.lgbm_importance_df.reset_index(drop=True, inplace=True)

        #Importance Plot
        if return_plot:
            UtilityFunctions.plot_importance(self.lgbm_importance_df)

        return self.lgbm_importance_df
    
    
    def xgb_importance(self, hyperparam_dict=None, objective=None, return_plot=False):
        """
        Calculate feature importance using XGBoost model.

        Parameters:
            hyperparam_dict (dict): Hyperparameters for the XGBoost model. (Optional)
            objective (str): Objective function for the XGBoost model. (Optional)
            return_plot (bool): Whether to return an importance plot.

        Returns:
            xgb_importance_df (pd.DataFrame): DataFrame with feature importances.
        """

        if hyperparam_dict == None:
            hyperparam_dict = FeatureSelectorConfig(seed=self.seed).xgb_hyperparams
        if objective is not None:
            hyperparam_dict['objective'] = objective

        #Model Training
        model = create_boosting_model(algorithm='xgboost',
                                      modeling_type=self.modeling_type,
                                      hyperparams=hyperparam_dict,
                                      numeric_columns=self.numeric_columns,
                                      categorical_columns=self.categorical_columns,
                                      preprocessing_params={'encoder':'te'})

        model.fit(X=self.X_train, y=self.y_train, model__sample_weight = self.weight_train)

        #Feature Importance DataFrame
        self.xgb_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'importance' : model.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.xgb_importance_df.reset_index(drop=True, inplace=True)

        #Importance Plot
        if return_plot:
            UtilityFunctions.plot_importance(self.xgb_importance_df)


        return self.xgb_importance_df
    
    
    def rf_importance(self, hyperparam_dict=None, return_plot=False):
        """
        Calculate feature importance using Random Forest model.

        Parameters:
            hyperparam_dict (dict): Hyperparameters for the Random Forest model. (Optional)
            return_plot (bool): Whether to return an importance plot.

        Returns:
            rf_importance_df (pd.DataFrame): DataFrame with feature importances.
        """

        if hyperparam_dict == None:
            hyperparam_dict = FeatureSelectorConfig(seed=self.seed).rf_hyperparams

        #Model Training
        preprocessing_pipeline = create_preporcessing_pipeline(numeric_columns=self.numeric_columns,
                                                              categorical_columns=self.categorical_columns, 
                                                              num_imputer='mean',
                                                              cat_imputer='constant',
                                                              cat_imp_value='unknown',
                                                              encoder='te')

        if self.modeling_type == 'regression':
            rf_model = RandomForestRegressor(**hyperparam_dict)
        elif self.modeling_type == 'classification':
            rf_model = RandomForestClassifier(**hyperparam_dict)

        model = Pipeline(steps=[('preprocessor', preprocessing_pipeline),
                                ('model', rf_model)])
        model.fit(X=self.X_train, y=self.y_train, model__sample_weight = self.weight_train)

        #Feature Importance DataFrame
        self.rf_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'importance' : model.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.rf_importance_df.reset_index(drop=True, inplace=True)

        #Importance Plot
        if return_plot:
            UtilityFunctions.plot_importance(self.rf_importance_df)

        return self.rf_importance_df
    
    
    def lassocv_coefs(self, hyperparam_dict=None, return_plot=False):
        """
        Calculate feature importance using LassoCV model.

        Parameters:
            hyperparam_dict (dict): Hyperparameters for the LassoCV model. (Optional)
            return_plot (bool): Whether to return an importance plot.

        Returns:
            lassocv_importance_df (pd.DataFrame): DataFrame with feature importances.
        """

        if hyperparam_dict == None:
            hyperparam_dict = FeatureSelectorConfig(seed=self.seed).lasso_hyperparams

        #Model Training
        preprocessing_pipeline = create_preporcessing_pipeline(numeric_columns=self.numeric_columns,
                                                              categorical_columns=self.categorical_columns, 
                                                              num_imputer='mean',
                                                              cat_imputer='constant',
                                                              cat_imp_value='unknown',
                                                              encoder='ohe')

        model = Pipeline(steps=[('preprocessor', preprocessing_pipeline),
                                ('model', LassoCV(**hyperparam_dict))])
        model.fit(X=self.X_train, y=self.y_train)

        #Feature Importance DataFrame
        feature_names = self.numeric_columns + model.named_steps['preprocessor']\
                                                        .named_transformers_["categorical_trans"]['encoder']\
                                                        .get_feature_names(self.categorical_columns).tolist()

        self.lasso_importance_df = pd.DataFrame({'feature':feature_names, 'importance':model['model'].coef_})

        self.lasso_importance_df['importance'] = abs(self.lasso_importance_df['importance'])

        mapping_dict={}
        for i in self.lasso_importance_df.feature.values[len(self.numeric_columns):]:
            temp_mapping_list = [x for x in self.categorical_columns if x in i]
            mapping_dict[i] = max(temp_mapping_list, key=len)

        for i in self.lasso_importance_df.feature.values[:len(self.numeric_columns)]:
            mapping_dict[i] = i

        self.lasso_importance_df['org_feature'] = [mapping_dict[x] for x in self.lasso_importance_df.feature]

        self.lasso_importance_df = self.lasso_importance_df.groupby('org_feature').importance.sum().to_frame()
        self.lasso_importance_df.reset_index(inplace=True, drop=False)
        self.lasso_importance_df.columns = ['feature', 'importance']
        self.lasso_importance_df.sort_values('importance', ascending=False, inplace=True)
        
        self.lasso_importance_df.reset_index(drop=True, inplace=True)

        #Importance Plot
        if return_plot:
            UtilityFunctions.plot_importance(self.lasso_importance_df)

        return self.lasso_importance_df
    
    
    def permutation_importance(self, lgbm_hyperparams=None, perimp_hyperparams=None, return_plot=False):
        """
        Calculate feature importance using permutation importance.

        Parameters:
            lgbm_hyperparams (dict): Hyperparameters for the LightGBM model. (Optional)
            perimp_hyperparams (dict): Hyperparameters for permutation importance. (Optional)
            return_plot (bool): Whether to return an importance plot.

        Returns:
            perimp_importance_df (pd.DataFrame): DataFrame with feature importances.
        """

        if lgbm_hyperparams == None:
            lgbm_hyperparams = FeatureSelectorConfig(seed=self.seed).lgbm_hyperparams

        if perimp_hyperparams == None:
            perimp_hyperparams = FeatureSelectorConfig(seed=self.seed).perimp_hyperparams

        #Permutation Importance Training
        model = create_boosting_model(algorithm='lightgbm',
                          modeling_type=self.modeling_type,
                          hyperparams=lgbm_hyperparams)

        model.fit(X=self.X_train, y=self.y_train, sample_weight=self.weight_train)

        per_imp = permutation_importance(estimator=model, X=self.X_test, y=self.y_test)

        self.perimp_importance_df = pd.DataFrame({'feature':self.X_train.columns,
                                                  'importance':list(per_imp.importances_mean)})

        self.perimp_importance_df.sort_values('importance', ascending=False, inplace=True)
        
        self.perimp_importance_df.reset_index(drop=True, inplace=True)

        #Importance Plot
        if return_plot:
            UtilityFunctions.plot_importance(self.perimp_importance_df)

        return self.perimp_importance_df
    
    
    def rfecv_importance(self, lgbm_hyperparams=None, rfecv_hyperparams=None, return_plot=False):
        """
        Calculate feature importance using RFECV (Recursive Feature Elimination with Cross-Validation).

        Parameters:
            lgbm_hyperparams (dict): Hyperparameters for the LightGBM model. (Optional)
            rfecv_hyperparams (dict): Hyperparameters for RFECV. (Optional)
            return_plot (bool): Whether to return an importance plot.

        Returns:
            rfecv_importance_df (pd.DataFrame): DataFrame with feature rankings.
        """

        if lgbm_hyperparams == None:
            lgbm_hyperparams = FeatureSelectorConfig(seed=self.seed).lgbm_hyperparams

        if rfecv_hyperparams == None:
            rfecv_hyperparams = FeatureSelectorConfig(seed=self.seed).rfecv_hyperparams

        preprocessing_pipeline = create_preporcessing_pipeline(numeric_columns=self.numeric_columns,
                                                          categorical_columns=self.categorical_columns,
                                                          encoder='te')

        X_train_te = preprocessing_pipeline.fit_transform(X=self.X_train, y=self.y_train)

        #RFECV Training
        model = create_boosting_model(algorithm='lightgbm',
                          modeling_type=self.modeling_type,
                          hyperparams=lgbm_hyperparams)

        rfe = RFECV(model, **rfecv_hyperparams)
        rfe.fit(X_train_te, self.y_train)

        self.rfe_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'importance' : rfe.ranking_
        }).sort_values('importance', ascending=True)
        
        self.rfe_importance_df.reset_index(drop=True, inplace=True)

        #Importance Plot
        if return_plot:
            UtilityFunctions.plot_importance(self.rfe_importance_df)

        return self.rfe_importance_df
    
    
    def boruta_rankings(self, rf_hyperparams=None, boruta_hyperparams=None):
        """
        Calculate feature importance using Boruta feature selection.

        Parameters:
            rf_hyperparams (dict): Hyperparameters for the Random Forest model. (Optional)
            boruta_hyperparams (dict): Hyperparameters for Boruta. (Optional)
        
        Returns:
            boruta_importance_df (pd.DataFrame): DataFrame with feature importances.
        """

        if rf_hyperparams == None:
            rf_hyperparams = FeatureSelectorConfig(seed=self.seed).rf_hyperparams

        if boruta_hyperparams == None:
            boruta_hyperparams = FeatureSelectorConfig(seed=self.seed).boruta_hyperparams


        #Model Training
        preprocessing_pipeline = create_preporcessing_pipeline(numeric_columns=self.numeric_columns,
                                                              categorical_columns=self.categorical_columns, 
                                                              num_imputer='mean',
                                                              cat_imputer='constant',
                                                              cat_imp_value='unknown',
                                                              encoder='te')

        if self.modeling_type == 'regression':
            model = RandomForestRegressor(**rf_hyperparams)
        elif self.modeling_type == 'classification':
            model = RandomForestClassifier(**rf_hyperparams)

        X_trans = preprocessing_pipeline.fit_transform(self.X_train, self.y_train)

        boruta = BorutaPy(estimator = model, **boruta_hyperparams)

        boruta.fit(np.array(X_trans), np.array(self.y_train))

        #Feature Importance DataFrame
        self.boruta_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'boruta_support' : boruta.support_ * 1,
            'boruta_support_weak' : boruta.support_weak_ * 1,
            'boruta_ranking' : boruta.ranking_
        }).sort_values('boruta_ranking', ascending=True)

        self.boruta_importance_df.reset_index(drop=True, inplace=True)

        return self.boruta_importance_df

class AutoFeatureSelect(FeatureSelector):
    """
    Class for automatic feature selection based on various importance methods.
    Inherits from FeatureSelector class.
    """
    
    def calculate_correlated_features(self, static_features=None, num_corr_method='pearson', num_threshold=0.9, cat_threshold=0.9):
        """
        Calculate correlated features based on numeric and categorical correlations.

        Parameters:
        -----------
        static_features : List[str], optional (default=None)
            List of feature names that are considered static and should not be removed.

        num_corr_method : str (default='pearson')
            Method to compute numeric correlations (e.g., 'pearson', 'spearman', 'kendall').

        num_threshold : float (default=0.9)
            Correlation threshold for numeric features.

        cat_threshold : float (default=0.9)
            Correlation threshold for categorical features.

        Returns:
        --------
        corr_features : List[str]
            List of correlated features that may be removed.
            
        Examples:
        ---------
        auto_selector = AutoFeatureSelect(modeling_type='regression', X_train=X_train, y_train=y_train,
                                          numeric_columns=num_features, categorical_columns=cat_features, seed=42)
        corr_features = auto_selector.calculate_correlated_features()
        auto_selector.drop_correlated_features()
        """
            
        if self.numeric_columns is not None:
            
            if static_features is not None:
                num_static_features = [i for i in static_features if i in self.numeric_columns]
            else:
                num_static_features = None
            
            _, num_corr_list = CorrelationCalculator.numeric_correlations(X=self.X_train,
                                                                          features=self.numeric_columns,
                                                                          static_features=num_static_features,
                                                                          corr_method=num_corr_method,
                                                                          threshold=num_threshold
                                                                        )
        else:
            num_corr_list = []
            
        if self.categorical_columns is not None:
            
            if static_features is not None:
                cat_static_features = [i for i in static_features if i in self.categorical_columns]
            else:
                cat_static_features = None
        
            _, cat_corr_list = CorrelationCalculator.categorical_correlations(X=self.X_train,
                                                                          features=self.categorical_columns,
                                                                          static_features=cat_static_features,
                                                                          threshold=cat_threshold
                                                                        )
        else:
            cat_corr_list = []
        
        self.corr_features = num_corr_list + cat_corr_list
        
        return self.corr_features
    
    def drop_correlated_features(self):
        """
        Drop correlated features from the dataset.
        
        Examples:
        ---------
        auto_selector = AutoFeatureSelect(modeling_type='regression', X_train=X_train, y_train=y_train,
                                          numeric_columns=num_features, categorical_columns=cat_features, seed=42)
        corr_features = auto_selector.calculate_correlated_features()
        auto_selector.drop_correlated_features()
        """
        
        if len(self.corr_features) > 0:
            self.X_train.drop(self.corr_features, axis=1, inplace=True)
            if self.X_test is not None:
                self.X_test.drop(self.corr_features, axis=1, inplace=True)

            if self.numeric_columns is not None:
                self.numeric_columns = [i for i in self.numeric_columns if i not in self.corr_features]
            if self.categorical_columns is not None:
                self.categorical_columns = [i for i in self.categorical_columns if i not in self.corr_features]
    
    
    def apply_feature_selection(self, selection_methods=['lgbm', 'perimp', 'rfecv', 'boruta'],
                                lgbm_hyperparams=None, xgb_hyperparams=None, rf_hyperparams=None,
                                lassocv_hyperparams=None, perimp_hyperparams=None,
                                rfecv_hyperparams=None, boruta_hyperparams=None):
        """
        Apply multiple feature selection methods and combine their results.
        
        Parameters:
        -----------
        selection_methods : List[str]
            List of feature selection methods to apply.

        lgbm_hyperparams : dict, optional (default=None)
            Hyperparameters for LightGBM model.

        xgb_hyperparams : dict, optional (default=None)
            Hyperparameters for XGBoost model.

        rf_hyperparams : dict, optional (default=None)
            Hyperparameters for RandomForest model.

        lasso_hyperparams : dict, optional (default=None)
            Hyperparameters for LassoCV model.

        perimp_hyperparams : dict, optional (default=None)
            Hyperparameters for permutation importance calculation.

        rfecv_hyperparams : dict, optional (default=None)
            Hyperparameters for Recursive Feature Elimination with Cross-Validation.

        boruta_hyperparams : dict, optional (default=None)
            Hyperparameters for Boruta feature selection.

        Returns:
        --------
        final_importance_df : pandas.DataFrame
            Combined feature importance results.

        Examples:
        ---------
        auto_selector = AutoFeatureSelect(modeling_type='regression', X_train=X_train, y_train=y_train,
                                          numeric_columns=num_features, categorical_columns=cat_features, seed=42)
                                          
        final_importance_df = auto_selector.apply_feature_selection(selection_methods=['lgbm', 'rfecv'],
                                                                     lgbm_hyperparams=lgbm_params,
                                                                     rfecv_hyperparams=rfecv_params)

        """
        
        
        dataframes, column_names = [], ['feature']
        if 'lgbm' in selection_methods:
            try:
                lgbm_importance_df = self.lgbm_importance(hyperparam_dict=lgbm_hyperparams)
                lgbm_importance_df.columns = ['feature', 'lgbm_importance']
                dataframes.append(lgbm_importance_df)
                column_names.append('lgbm_importance')
                print("LightGBM Feature Importance is finished")
            except Exception as e:
                print(f"An error occurred: {e}")

        if 'xgb' in selection_methods:
            try:
                xgb_importance_df = self.xgb_importance(hyperparam_dict=xgb_hyperparams)
                xgb_importance_df.columns = ['feature', 'xgb_importance']
                dataframes.append(xgb_importance_df)
                column_names.append('xgb_importance')
                print("XGBoost Feature Importance is finished")
            except Exception as e:
                print(f"An error occurred: {e}")

        if 'rf' in selection_methods:
            try:
                rf_importance_df = self.rf_importance(hyperparam_dict=rf_hyperparams)
                rf_importance_df.columns = ['feature', 'rf_importance']
                dataframes.append(rf_importance_df)
                column_names.append('rf_importance')
                print("Random Forest Feature Importance is finished")
            except Exception as e:
                print(f"An error occurred: {e}")

        if 'lassocv' in selection_methods:
            try:
                lassocv_importance_df = self.lassocv_importance(hyperparam_dict=lassocv_hyperparams)
                lassocv_importance_df.columns = ['feature', 'lassocv_coefs']
                dataframes.append(lassocv_importance_df)
                column_names.append('lassocv_coefs')
                print("LassoCV Feature Importance is finished")
            except Exception as e:
                print(f"An error occurred: {e}")

        if 'perimp' in selection_methods:
            try:
                perimp_importance_df = self.permutation_importance(lgbm_hyperparams=lgbm_hyperparams, perimp_hyperparams=perimp_hyperparams)
                perimp_importance_df.columns = ['feature', 'permutation_importance']
                dataframes.append(perimp_importance_df)
                column_names.append('permutation_importance')
                print("Permutation Importance Feature Importance is finished")
            except Exception as e:
                print(f"An error occurred: {e}")

        if 'rfecv' in selection_methods:
            try:
                rfecv_importance_df = self.rfecv_importance(lgbm_hyperparams=lgbm_hyperparams, rfecv_hyperparams=rfecv_hyperparams)
                rfecv_importance_df.columns = ['feature', 'rfecv_rankings']
                dataframes.append(rfecv_importance_df)
                column_names.append('rfecv_rankings')
                print("RFECV Feature Importance is finished")
            except Exception as e:
                print(f"An error occurred: {e}")
                
        if 'boruta' in selection_methods:
            try:
                boruta_importance_df = self.boruta_rankings(rf_hyperparams=rf_hyperparams, boruta_hyperparams=boruta_hyperparams)
                boruta_importance_df.columns = ['feature', 'boruta_support', 'boruta_support_weak', 'boruta_ranking']
                dataframes.append(boruta_importance_df)
                column_names = column_names + ['boruta_support', 'boruta_support_weak', 'boruta_ranking']
                print("Boruta Feature Importance is finished")
            except Exception as e:
                print(f"An error occurred: {e}")


        self.final_importance_df = reduce(lambda  left,right: pd.merge(left,right,on=['feature'], how='outer'),
                                     dataframes)

        #self.final_importance_df.columns = column_names

        return self.final_importance_df