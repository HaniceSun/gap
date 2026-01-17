import os
import numpy as np
import pandas as pd
import yaml
import ast
import joblib
import time
import datetime
from zoneinfo import ZoneInfo
from itertools import product
from importlib import resources
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import CalibratedClassifierCV


class GapLearn():
    def __init__(self, task_type='classification', multi_class=True, score_method='roc_auc'):
        self.time_zone = ZoneInfo("America/Los_Angeles")
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.config_dir = f'{resources.files("gap").parent}/config'
        self.score_methods = {'roc_auc':roc_auc_score, 'mse':mean_squared_error, 'mae':mean_absolute_error, 'r2':r2_score}
        self.task_type = task_type
        self.multi_class = multi_class
        self.score_method = score_method

    def train_cross_val(self, config_file='config.yaml', train_file='features_labeled_train.txt',
                        metrics_file='metrics.txt', n_features='max', target_col='class',
                        exclude_columns=['SampleID', 'class_name'], cv_n_splits=10,
                        feature_standardize=False, feature_select=False, random_state=42):

        self.config = self.load_yaml(config_file)
        self.df_train = pd.read_table(train_file, header=0, sep='\t')
        self.target_col = target_col
        self.feature_cols = [True if x not in [self.target_col] + exclude_columns else False for x in self.df_train.columns]
        self.log(f'train shape: {self.df_train.shape}')

        self.cvs = KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)

        try:
            n_features = [x.strip() for x in n_features.split(',')]
            self.n_features = [int(x) for x in n_features if x.isdigit()] + [x for x in n_features if not x.isdigit()]
        except Exception as e:
            print(f'ERROR parsing n_features {e}')

        idx_splits = -1
        for train_idx, val_idx in self.cvs.split(self.df_train):
            idx_splits += 1
            X_train = self.df_train.loc[train_idx, self.feature_cols]
            y_train = self.df_train.loc[train_idx, self.target_col]
            X_val = self.df_train.loc[val_idx, self.feature_cols]
            y_val = self.df_train.loc[val_idx, self.target_col]
            if feature_select:
                features = self.feature_selection(X_train, y_train)
            else:
                features = list(X_train.columns)
            X_train_selected = X_train.loc[:, features]
            X_val_selected = X_val.loc[:, features]
            if feature_standardize:
                X_train_selected, X_val_selected, scaler = self.feature_standardization(X_train_selected, X_val_selected)
            for nf in self.n_features:
                if nf == 'max':
                    nf = X_train_selected.shape[1]
                X_train = X_train_selected.iloc[:, 0:nf]
                X_val = X_val_selected.iloc[:, 0:nf]
                for model_name, model_info in self.config['models'].items():
                    model_class = eval(model_info['class'])
                    param_grid = model_info['params']
                    keys = list(param_grid.keys())
                    combinations = list(product(*param_grid.values()))
                    for combo in combinations:
                        params = dict(zip(keys, combo))
                        model = self.fit_model(model_name, model_class, params, X_train, y_train, X_val, y_val, random_state)
                        y_pred = model.predict(X_val)
                        if self.task_type == 'classification':
                            y_pred_prob = model.predict_proba(X_val)
                            if self.multi_class:
                                score = self.score_methods[self.score_method](y_val, y_pred_prob, multi_class='ovr', average='macro')
                                extra = classification_report(y_val, y_pred, output_dict=True)
                            else:
                                score = self.score_methods[self.score_method](y_val, y_pred_prob[:, 1])
                                extra = confusion_matrix(y_val, y_pred).ravel()
                                extra = ','.join([str(x) for x in extra])
                        elif self.task_type == 'regression':
                            score = self.score_methods[self.score_method](y_val, y_pred)
                            extra = '.'
                        self.log_metrics([model_name, str(nf), str(params), idx_splits] + [self.score_method, score, 'val', extra], out_file=metrics_file)

    def feature_selection(self, X_train, y_train, remove_low_variance=True, remove_high_correlation=True, sort_by_mutual_info_score=True):
        if remove_low_variance:
            X_train = self._remove_low_variance(X_train)
        if remove_high_correlation:
            X_train = self._remove_high_correlation(X_train)
        if sort_by_mutual_info_score:
            features = self._sort_by_mutual_info_score(X_train, y_train)
        else:
            features = list(X_train.columns)
        return features

    def feature_standardization(self, X, X2=None, scaler=None):
        if not scaler:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            if X2 is not None:
                X2 = pd.DataFrame(scaler.transform(X2), columns=X2.columns)
        else:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)
            if X2 is not None:
                X2 = pd.DataFrame(scaler.transform(X2), columns=X2.columns)
        return X, X2, scaler

    def fit_model(self, model_name, model_class, params, X_train, y_train, X_val=None, y_val=None, random_state=42):
        if model_name == 'mlp':
            model = model_class(**params, random_state=random_state, verbose=0)
            model.fit(X_train, y_train)
        elif model_name == 'catboost':
            model = model_class(**params, random_state=random_state, verbose=0)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val), logging_level='Silent')
            else:
                model.fit(X_train, y_train, logging_level='Silent')
        elif model_name == 'lgb':
            model = model_class(**params, random_state=random_state, verbose=-1)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
            else:
                model.fit(X_train, y_train)
        elif model_name == 'xgb':
            model = model_class(**params, random_state=random_state)
            if y_val is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
        elif model_name == 'random_forest':
            model = model_class(**params, random_state=random_state, verbose=0)
            model.fit(X_train, y_train)
        elif model_name == 'svm':
            model = model_class(**params, probability=True, random_state=random_state, verbose=0)
            model.fit(X_train, y_train)
        else:
            model = model_class(**params, random_state=random_state)
            model.fit(X_train, y_train)
        return model

    def get_best_model_and_params(self, metrics_file='metrics.txt', dataset='val'):
        self.metrics = pd.read_table(metrics_file, header=0, sep='\t')
        if self.metrics.shape[0]:
            L = []
            df1 = self.metrics.loc[self.metrics['dataset'] == dataset, ]
            for n_features in df1['n_features'].unique():
                df2 = df1.loc[df1['n_features'] == n_features, ]
                for model in df2['model'].unique():
                    df3 = df2.loc[df2['model'] == model, ]
                    for params in df3['params'].unique():
                        df4 = df3.loc[df3['params'] == params, ]
                        if self.task_type == 'classification':
                            if self.multi_class:
                                extra = '.'
                            else:
                                extra = df4['extra'].str.split(',', expand=True).astype(int).sum(axis=0)
                                extra = ','.join([str(x) for x in extra])
                        elif self.task_type == 'regression':
                            extra = '.'
                        sm = np.mean(df4['score'])
                        L.append([n_features, model, params] + [self.score_method, sm, dataset, extra])

            out_file = f'{metrics_file.replace(".txt", "_sorted.txt")}'
            out_file2 = f'{metrics_file.replace(".txt", "_sorted_best.txt")}'
            df = pd.DataFrame(L)
            df.columns = ['n_features', 'model', 'params'] + ['score_method', 'score', 'dataset', 'extra']
            if self.score_method in ['roc_auc', 'r2']:
                df.sort_values(by='score', ascending=False, inplace=True)
            elif self.score_method in ['mse', 'mae']:
                df.sort_values(by='score', ascending=True, inplace=True)
            df.to_csv(out_file, header=True, index=False, sep='\t')

            n_features = df['n_features'].iloc[0]
            df2 = df.loc[df['n_features'] == n_features, ]
            df3 = df2.drop_duplicates(subset=['model'], keep='first')
            df3.to_csv(out_file2, header=True, index=False, sep='\t')
            self.log('best model params:')
            print(df3)

    def final_fit_eval_on_full_train_then_eval_on_test(self, config_file='config.yaml', metrics_file='metrics_sorted_best.txt', train_file='features_labeled_train.txt', 
                                                       test_file='features_labeled_test.txt', exclude_columns=['SampleID', 'class_name'], target_col='class', 
                                                       feature_standardize=False, feature_select=False, ensemble=False, ensemble_voting='soft', 
                                                       ensemble_calibration=False, calibration_method='sigmoid', calibration_cv=5, model_file='model.pkl', random_state=42):
        self.config = self.load_yaml(config_file)
        df = pd.read_table(metrics_file, header=0, sep='\t')
        self.df_train = pd.read_table(train_file, header=0, sep='\t')
        self.df_test = pd.read_table(test_file, header=0, sep='\t')
        self.target_col = target_col
        self.feature_cols = [True if x not in [self.target_col] + exclude_columns else False for x in self.df_train.columns]

        to_save = {}
        X_train_full = self.df_train.loc[:, self.feature_cols]
        y_train = self.df_train.loc[:, self.target_col]
        X_test_full = self.df_test.loc[:, self.feature_cols]
        y_test = self.df_test.loc[:, self.target_col]
        if feature_select:
            features_full = self.feature_selection(X_train_full, y_train)
        else:
            features_full = list(X_train_full.columns)

        models = {}
        out_file = metrics_file.replace('.txt', '_eval.txt')
        self.log('final eval on full train and test:')

        for n in range(df.shape[0]):
            if ensemble:
                n_features = df['n_features'].iloc[0]
            else:
                n_features = df['n_features'].iloc[n]
            features = features_full[0:n_features]
            X_train = X_train_full.loc[:, features]
            X_test = X_test_full.loc[:, features]

            if feature_standardize:
                X_train, X_test, scaler = self.feature_standardization(X_train, X_test)
            else:
                scaler = None

            params = ast.literal_eval(df['params'].iloc[n])
            model_name = df['model'].iloc[n]
            model_class = eval(self.config['models'][model_name]['class'])

            model = self.fit_model(model_name, model_class, params, X_train, y_train, random_state=random_state)
            models[model_name] = model
            self.eval_model(model, X_train, y_train, model_name, n_features, 'train', out_file)
            self.eval_model(model, X_test, y_test, model_name, n_features, 'test', out_file)

            to_save[model_name] = model
            to_save[f'{model_name}_n_features'] = n_features
            to_save[f'{model_name}_features'] = features
            to_save[f'{model_name}_scaler'] = scaler

        if ensemble:
            model_name = 'ensemble'
            if ensemble_calibration:
                if self.task_type == 'classification':
                    for k in models:
                        models[k] = CalibratedClassifierCV(models[k], method=calibration_method, cv=calibration_cv)
                elif self.task_type == 'regression':
                    for k in models:
                        models[k] = CalibratedRegressorCV(models[k], method=calibration_method, cv=calibration_cv)

            if self.task_type == 'classification':
                model = VotingClassifier(estimators=list(models.items()), voting=ensemble_voting).fit(X_train, y_train)
            elif self.task_type == 'regression':
                model = VotingRegressor(estimators=list(models.items()), voting=ensemble_voting).fit(X_train, y_train)

            self.eval_model(model, X_train, y_train, model_name, n_features, 'train', out_file)
            self.eval_model(model, X_test, y_test, model_name, n_features, 'test', out_file)
            to_save[model_name] = model
            to_save[f'{model_name}_n_features'] = n_features
            to_save[f'{model_name}_features'] = features
            to_save[f'{model_name}_scaler'] = scaler

        joblib.dump(to_save, model_file)

    def predict(self, in_file='to_predict_features.txt', out_file='predicted.txt', label_file='features_labels.txt', model_name='random_forrest', model_file='model.pkl', feature_standardize=False, include_columns=['SampleID']):
        try:
            df_labels = pd.read_table(label_file, header=0, sep='\t')
            num2name = dict(zip(df_labels['class'], df_labels['class_name']))
        except Exception as e:
            print(f'ERROR loading label file {e}')
            num2name = {}

        M = joblib.load(model_file)
        features = list(M[f'{model_name}_features'])
        scaler = M[f'{model_name}_scaler']
        model = M[model_name]

        df = pd.read_table(in_file, header=0, sep='\t')
        X = df.loc[:, features]

        if feature_standardize:
            X, _, _ = self.feature_standardization(X, scaler=scaler)

        if list(X.columns) != list(features):
            self.log('ERROR features are different')

        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)

        df_y_pred = pd.DataFrame(y_pred)
        df_y_pred.columns = ['class']
        df_y_pred_prob = pd.DataFrame(y_pred_prob)

        if include_columns:
            df_res = pd.concat([df[include_columns], df_y_pred, df_y_pred_prob], axis=1)         
        else:
            df_res = pd.concat([df_y_pred, df_y_pred_prob], axis=1)

        try:
            df_res['class'] = df_res['class'].map(num2name)
            df_res.columns = ['SampleID', 'class'] + [num2name.get(x, '.') for x in df_res.columns[2:]]
        except Exception as e:
            print(f'ERROR mapping class names {e}')

        df_res.to_csv(out_file, header=True, index=False, sep='\t')

        self.log('final predicted results:')
        print(df_res.tail(10))

    def eval_model(self, model, X, y, model_name, n_features, dataset, out_file='eval.txt'):
        y_pred = model.predict(X)
        if self.task_type == 'classification':
            y_pred_prob = model.predict_proba(X)
            if self.multi_class:
                score = self.score_methods[self.score_method](y, y_pred_prob, multi_class='ovr', average='macro')
                extra = classification_report(y, y_pred, output_dict=True)
            else:
                score = self.score_methods[self.score_method](y, y_pred_prob[:, 1])
                extra = confusion_matrix(y, y_pred).ravel()
                extra = ','.join([str(x) for x in extra])
        elif self.task_type == 'regression':
            score = self.score_methods[score_method](y, y_pred)
            extra = '.'
        self.log_metrics([model_name, n_features, '.', '.', self.score_method, score, dataset, extra], out_file=out_file)

    def _remove_low_variance(self, X, threshold=1e-3):
        selector = VarianceThreshold(threshold=threshold)
        X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
        return X

    def _remove_high_correlation(self, X, threshold=0.95):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
        X.drop(columns=to_drop, inplace=True)
        return X

    def _sort_by_mutual_info_score(self, X, y, discrete_features='auto', random_state=42):
        mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)
        df = pd.DataFrame()
        df['features'] = X.columns
        df['mi_score'] = mi
        df.sort_values(by='mi_score', ascending=False, inplace=True)
        return list(df['features'])

    def log_metrics(self, L, out_file='metrics.txt'):
        current_time = datetime.datetime.now().astimezone(self.time_zone).strftime(self.time_format)
        line = '\t'.join([current_time] + [str(x) for x in L])
        self.log(line)

        header = ['time', 'model', 'n_features', 'params', 'idx_splits', 'score_method', 'score', 'dataset', 'extra']

        if out_file not in vars(self):
            setattr(self, out_file, True)
            with open(out_file, 'w') as f:
                f.write('\t'.join(header) + '\n')
                f.write(line + '\n')
        else:
            with open(out_file, 'a') as f:
                f.write(line + '\n')

    def log(self, txt=''):
        current_time = datetime.datetime.now().astimezone(self.time_zone).strftime(self.time_format)
        print(f'{current_time} {txt}', flush=True)

    def load_yaml(self, config_file):
        config = {}
        if not os.path.exists(config_file):
            config_file = self.config_dir + '/' + config_file
        config_file = os.path.expanduser(config_file)
        self.log(f'using config {config_file}')

        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.log(f'ERROR loading config {e}')
        return config

if __name__ == '__main__':
    gl = GapLearn()
    gl.train_cross_val()
    gl.get_best_model_and_params()
    gl.final_fit_eval_on_full_train_then_eval_on_test()
    gl.predict()
