import pandas as pd
import numpy as np
import random
import os
import sys
import yaml
from sklearn import datasets
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from dataclasses import dataclass
np.random.seed(0)
random.seed(0)

class Config:
    def __init__(self, config_yaml=None):

        self.models = {}   
        self.models['LR'] = LogisticRegression()
        self.models['KNN'] = KNeighborsClassifier()
        self.models['GNB'] = GaussianNB()
        self.models['SVM'] = SVC()
        self.models['RF'] = RandomForestClassifier(random_state=0)
        self.models['RFb'] = RandomForestClassifier(class_weight='balanced', random_state=0)
        self.models['XGB'] = XGBClassifier()

        self.config = str(config_yaml).split('.yaml')[0]
        if config_yaml:
            with open(config_yaml) as F:
                D = yaml.safe_load(F)
                vars(self).update(D)
        self.vars = vars(self)

        if 'binary_class' in self.vars and self.binary_class == True:
            self.scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
            self.scoring_fn = [precision_score, recall_score, accuracy_score, r2_score, confusion_matrix]
            self.confusion = ['TN', 'FP', 'FN', 'TP']
        elif 'multi_class' in self.vars and self.multi_class == True:
            self.scoring = ['f1_weighted']
            self.scoring_fn = [classification_report]

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metrics = {}

    def confusion_matrix_scorer(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred).ravel()
        return {'TN': cm[0], 'FP': cm[1], 'FN': cm[2], 'TP': cm[3]}

    def read_data(self):
        df = pd.read_table(self.cfg.inF, header=0, sep='\t')
        df_obs = df.loc[df['ObsPre'] == 'Observed', ]
        df_pred =  df.loc[(df['ObsPre'] == 'ToBePredicted') & ~(df['Sample'].str.startswith('HG') | df['Sample'].str.startswith('NA')), :]
        pcs = ['X' + str(x) for x in eval(self.cfg.pcs)]
        self.X = df_obs.loc[:, pcs]
        y = df_obs.loc[:, self.cfg.y_class]
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(y)
        self.X_pred = df_pred.loc[:, pcs]
        self.sample_pred = df_pred['Sample']

    def data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.cfg.train_size)

    def feature_selection(self, model_name='LR', selection_method='KBest'):
        if selection_method == 'KBest':
            fs = SelectKBest(f_classif, k=self.cfg.KBest_k)
            ft = fs.fit(self.X_train, self.y_train)
            self.X_train_new = ft.transform(self.X_train)
            self.X_test_new = ft.transform(self.X_test)
            self.X_pred_new = ft.transform(self.X_pred)
        elif selection_method == 'RFE':
            fs = RFE(self.cfg.models[model_name], n_features_to_select=self.cfg.RFE_n_features_to_select, step=1)
            ft = fs.fit(self.X_train, self.y_train)
            self.X_train_new = ft.transform(self.X_train)
            self.X_test_new = ft.transform(self.X_test)
            self.X_pred_new = ft.transform(self.X_pred)
        elif selection_method == 'RFECV':
            cv = model_selection.KFold(n_splits=self.cfg.RFECV_n_splits, shuffle=True)
            fs = RFECV(self.cfg.models[model_name], step=1, cv=cv, min_features_to_select=1, n_jobs=2)
            ft = fs.fit(self.X_train, self.y_train)
            self.X_train_new = ft.transform(self.X_train)
            self.X_test_new = ft.transform(self.X_test)
            self.X_pred_new = ft.transform(self.X_pred)
        elif selection_method == 'SFM':
            fit = self.cfg.models[model_name].fit(self.X_train, self.y_train)
            ft = SelectFromModel(fit, prefit=True)
            self.X_train_new = ft.transform(self.X_train)
            self.X_test_new = ft.transform(self.X_test)
            self.X_pred_new = ft.transform(self.X_pred)
        elif selection_method == 'PCA':
            fs = PCA(self.cfg.PCA_n_component)
            ft = fs.fit(self.X_train)
            self.X_train_new = ft.transform(self.X_train)
            self.X_test_new = ft.transform(self.X_test)
            self.X_pred_new = ft.transform(self.X_pred)
        elif selection_method == 'None':
            self.X_train_new = self.X_train
            self.X_test_new = self.X_test
            self.X_pred_new = self.X_pred

    def train_validate(self, model_name='LR', selection_method='KBest'):
        cv = model_selection.KFold(n_splits=self.cfg.model_selection_n_splits, shuffle=True)

        res = model_selection.cross_validate(self.cfg.models[model_name], self.X_train_new, self.y_train, cv=cv, scoring=self.cfg.scoring)
        if self.cfg.confusion_matrix:
            res2 = model_selection.cross_validate(self.cfg.models[model_name], self.X_train_new, self.y_train, cv=cv, scoring=Trainer.confusion_matrix_scorer)

        for s in self.cfg.scoring:
            m = np.mean(res[f'test_{s}'])
            self.log_metrics(model_name, selection_method, s, m, 'val')

        if self.cfg.confusion_matrix:
            m = [np.mean(res2[f'test_{s}']) for s in self.cfg.confusion]
            self.log_metrics(model_name, selection_method, 'confusion_matrix', ','.join([str(x) for x in m]), 'val')

    def model_test(self, model_name='LR', selection_method='KBest'):
        self.feature_selection(model_name, selection_method)
        ft = self.cfg.models[model_name].fit(self.X_train_new, self.y_train)
        self.y_train_pred = ft.predict(self.X_train_new)
        self.y_test_pred = ft.predict(self.X_test_new)
        for s in self.cfg.scoring_fn:
            m_train = s(self.y_train, self.y_train_pred)
            m_test = s(self.y_test, self.y_test_pred)
            if s.__name__ == 'confusion_matrix':
                m_train = ','.join([str(x) for x in m_train.ravel()])
                m_test = ','.join([str(x) for x in m_test.ravel()])
            elif s.__name__ == 'classification_report':
                print('train dataset classification_report:')
                print(m_train)
                print('test dataset classification_report:')
                print(m_test)

            self.log_metrics(model_name, selection_method, s.__name__, m_train, 'train')
            self.log_metrics(model_name, selection_method, s.__name__, m_test, 'test')
    
        if self.cfg.predict:
            y_pred = ft.predict(self.X_pred_new)
            y_prob = ft.predict_proba(self.X_pred_new)

            ouF = self.cfg.inF.split('.txt')[0] + f'_{model_name}_{self.cfg.y_class}.txt'
            df = pd.DataFrame(y_prob)
            df.columns = self.le.inverse_transform(ft.classes_)
            df['Sample'] = list(self.sample_pred)
            df['Class'] = self.le.inverse_transform(y_pred)
            df.to_csv(ouF, header=True, index=False, sep='\t', float_format='%.4f')
 

    def log_metrics(self, model_name, selection_method, score_method, score, dataset):
        self.metrics.setdefault('dataset', [])
        self.metrics.setdefault('model_name', [])
        self.metrics.setdefault('feature_selection', [])
        self.metrics.setdefault('score_method', [])
        self.metrics.setdefault('score', [])

        self.metrics['model_name'].append(model_name)
        self.metrics['feature_selection'].append(selection_method)
        self.metrics['score_method'].append(score_method)
        self.metrics['score'].append(score)
        self.metrics['dataset'].append(dataset)

    def save_metrics(self):
        self.metrics_df = pd.DataFrame(self.metrics)
        ouF = self.cfg.config + '_metrics.txt'
        if os.path.exists(ouF):
            self.metrics_df.to_csv(ouF, header=False, index=False, sep='\t', mode='a')
        else:
            self.metrics_df.to_csv(ouF, header=True, index=False, sep='\t')

    def __call__(self):
        self.read_data()
        self.data_split()

if __name__ == '__main__':

    config_yaml = None
    if len(sys.argv) > 1:
        config_yaml = sys.argv[1]
    cfg = Config(config_yaml)

    trainer = Trainer(cfg)
    trainer()

    if cfg.train:
        for model_name in cfg.train_model_names:
            for selection_method in cfg.train_feature_selection_methods:
                print([model_name, selection_method])
                trainer.feature_selection(model_name, selection_method)
                trainer.train_validate(model_name, selection_method)

    if cfg.test:
        trainer.model_test(cfg.test_model_name, cfg.test_feature_selection_method)

    trainer.save_metrics()

