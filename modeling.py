import stock as Stock
import numpy as np
import joblib as joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from alive_progress import alive_bar


class Modeling:
    def init(self):
        self.stock = Stock.Stock()
        self.ticker = ''
        
    def run(self, model = RandomForestClassifier, data = None):
        
        data = self.stock.get_stock_data(self.ticker, '10y')

        tscv = TimeSeriesSplit(n_splits = 10, max_train_size=364*2, test_size=90)
        
        precision = 0
        recall = 0
        roc_auc = 0
        
        for train_index, test_index in tscv.split(data):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            
            y_train = X_train['Target']
            y_test = X_test['Target']
            
            X_train.drop(columns=['Target'], inplace=True)
            X_test.drop(columns=['Target'], inplace=True)
            
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            y_pred = np.where(y_pred > 0.7, 1, 0)
            
            precision += precision_score(y_test, y_pred)
            recall += recall_score(y_test, y_pred)
            roc_auc += roc_auc_score(y_test, y_pred)
            
        return precision/10, recall/10, roc_auc/10, model
        
    def hyperparameter_tuning(self):

        hyperparameters = {
            'n_estimators': [100, 130],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'bootstrap': [True, False]
        }
        
        best_precision = 0
        best_recall = 0
        best_roc_auc = 0
        best_model = None
        
        best_params = {}
        
        with alive_bar(len(hyperparameters['n_estimators']) * len(hyperparameters['max_depth']) * len(hyperparameters['bootstrap'])) as bar:
            for n_estimators in hyperparameters['n_estimators']:
                for max_depth in hyperparameters['max_depth']:
                    for bootstrap in hyperparameters['bootstrap']:
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, random_state=42)
                        precision, recall, roc_auc, model = self.run(model)
                        if precision > best_precision and recall > best_recall:
                            best_precision = precision
                            best_recall = recall
                            best_roc_auc = roc_auc
                            best_model = model
                            best_params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'bootstrap': bootstrap
                            }
                            print(f'Best Precision: {best_precision}')
                            print(f'Best Recall: {best_recall}')
                            print(f'Best ROC AUC: {best_roc_auc}')
                            print(f'Best Params: {best_params}')
                        bar()
                        
        return best_precision, best_recall, best_roc_auc, best_model, best_params
                    

if __name__ == '__main__':
    model = Modeling()
    model.init()
    model.ticker = 'AMZN'
    best_precision, best_recall, best_roc_auc, best_model, best_params = model.hyperparameter_tuning()
    
    # save the model
    joblib.dump(best_model, 'model.pkl')
    
    model = joblib.load('model.pkl')
    
    feature_importance = model.feature_importances_
    feature_names = model.feature_names_in_
    
    feature_importance = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
    
    for feature in feature_importance:
        print(f'{feature[0]}: {feature[1]}')
    