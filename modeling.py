import stock as Stock
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from alive_progress import alive_bar


class Modeling:
    def init(self):
        self.stock = Stock.Stock()
        self.ticker = ''
        
    def train_test_split(self, data, train_size, validation_size):
        '''
        Split the data into train, test, and validation set
        '''
        
        train_set = data[:train_size]
        test_set = data[train_size:validation_size]
        validation_set = data[validation_size:]
        
        return train_set, test_set, validation_set
    
    def RF_model(self, train_set, n_estimators, max_depth, min_samples_split=2, min_samples_leaf=1):
        '''
        Random Forest model
        '''
        
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf,
            random_state=42
            )
        
        X_train = train_set.drop('Target', axis=1)
        y_train = train_set['Target']
        
        rf_model.fit(X_train, y_train)
        
        return rf_model
    
    def evaluate_model(self, model, validation_set, start=1000, step=90):
        '''
        Evaluate the model
        '''
        
        precision = []
        recall = []
        roc_auc = []
        
        with alive_bar(len(validation_set[start::step])) as bar:
            for i in range(start, len(validation_set)-1, step):
                X_validation = validation_set.drop('Target', axis=1)[:i]
                y_validation = validation_set['Target'][:i]
                
                y_pred = model.predict_proba(X_validation)
                y_pred = y_pred[:, 1]
                y_pred = np.where(y_pred > 0.6, 1, 0)
                
                precision.append(precision_score(y_validation, y_pred))
                recall.append(recall_score(y_validation, y_pred))
                roc_auc.append(roc_auc_score(y_validation, y_pred))
                
                bar()
                
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        roc_auc = sum(roc_auc) / len(roc_auc)
        
        return precision, recall, roc_auc
    
    def hyperparameter_tuning(self, train_set, test_set, start, step):
        hyperparameter = {
            'n_estimators': [10, 50, 100],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10, 30]
        }
        
        best_model = None
        best_roc_auc = 0
        
        with alive_bar(len(hyperparameter['n_estimators']) * len(hyperparameter['max_depth']) * len(hyperparameter['min_samples_split']) * len(hyperparameter['min_samples_leaf']) * len(test_set[start::step])) as bar:
            for n_estimators in hyperparameter['n_estimators']:
                for max_depth in hyperparameter['max_depth']:
                    for min_samples_split in hyperparameter['min_samples_split']:
                        for min_samples_leaf in hyperparameter['min_samples_leaf']:
                            
                            model = self.RF_model(train_set, n_estimators, max_depth, min_samples_split, min_samples_leaf)
                            
                            roc_auc_scores = []
                            
                            for i in range(start, len(test_set)-1, step):
                                X_test = test_set.drop('Target', axis=1)[:i]
                                y_test = test_set['Target'][:i]
                                
                                y_pred = model.predict_proba(X_test)[:, 1]
                                y_pred = np.where(y_pred > 0.6, 1, 0)
                                roc_auc_scores.append(roc_auc_score(y_test, y_pred))
                                
                                bar()
                                
                            roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
                                
                            if roc_auc > best_roc_auc:
                                best_roc_auc = roc_auc
                                best_model = model
                                best_hyperparameter = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf
                                }
                                
                                print(f'Best ROC AUC: {best_roc_auc}\nBest Hyperparameter: {best_hyperparameter}')
                            
                                
        return best_model, best_hyperparameter
    
    def run(self, ticker='AAPL', period='max', start=1000, step=90):
        self.init()
        self.ticker = 'AAPL'
        
        data = self.stock.get_stock_data(self.ticker, period)
        
        train_size = int(len(data) * 0.8)
        validation_size = int(len(data) * 0.9)
        
        print(train_size, validation_size, len(data))
        
        train_set, test_set, validation_set = self.train_test_split(data, train_size, validation_size)
        
        model, hyperparameter = self.hyperparameter_tuning(train_set, test_set, start, step)
        
        start = len(data) - validation_size*0.9
        step = 90
        
        precision, recall, roc_auc = self.evaluate_model(model, validation_set, start, step)
        
        print(f'Precision: {precision}\nRecall: {recall}\nROC AUC: {roc_auc}')
        
        return model, hyperparameter
    
    
if __name__ == '__main__':
    model = Modeling()
    model.run()