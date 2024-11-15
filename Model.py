import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow

#Data preprocessing
from sklearn.preprocessing import MinMaxScaler
#Model building
#Dealing with imbalanced datasets
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
#Check the model effect
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
#Gradient Boosting & XGBoost
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
#Adam & SGD neural networks
from sklearn.neural_network import MLPClassifier
#Using Adam and compare L2 regularisation and dropouts
from itertools import product
from tensorflow import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical

def load_data(dataset_name):
    if dataset_name == 'abalone':
        columns = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]
        data_in = pd.read_csv('./abalone_data/abalone.data', names = columns, delimiter = ',', header = None)
        #Classify the ring age into 4 groups by creating new category column
        data_in['category'] = pd.cut(data_in['Rings'], bins = [0, 8, 11, 15, float('inf')], labels = [0, 1, 2, 3])
    elif dataset_name == 'CMC':
        columns = ["wife_age", "wife_edu", "husband_edu", "num_children", "wife_religion", "wife_working", "husband_occupation", "standard_of_living_index", "media_exposure", "contraceptive_method"]
        data_in = pd.read_csv('./cmc_data/cmc.data', names = columns, delimiter = ',', header = None)
        #Remove the special characters
        characters = ['?', '$']
        data_in.replace(characters, '', regex = True, inplace = True)
    return data_in, columns

def data_visualisation(dataset_name, data_in, columns):
    if dataset_name == 'abalone':
        #Use catplot&countplot to report the distribution of class 
        plt.figure(figsize = (10, 10))
        sns.catplot(data = data_in, x = 'category', y = 'Rings')
        plt.title('abalone_class_distribution')
        plt.savefig('./abalone_results/abalone_class_distribution.png')
        plt.clf()
        plt.figure(figsize = (10, 10))
        sns.countplot(data = data_in, x = 'category')
        plt.title('abalone_class_count')
        plt.savefig('./abalone_results/abalone_class_count.png')
        plt.clf()
        #Use pairplot to report the distribution of features in pairs
        data_features = data_in.drop(columns = ['Rings', 'category'])
        data_features['Sex'] = np.where(data_features['Sex'] == 'M', 1, np.where(data_features['Sex'] == 'F', 2, 0))
        plt.figure(figsize = (30,30))
        sns.pairplot(data = data_features, kind = 'scatter', diag_kind = 'kde')
        plt.title('abalone_features_distribution')
        plt.savefig('./abalone_results/abalone_features_distribution.png')
        plt.clf()
        #Correlation heatmap
        corr = data_features.corr(numeric_only = True)
        plt.figure(figsize = (10,10))
        sns.heatmap(corr, vmin = -1, vmax = 1, annot = True, cmap = 'coolwarm')
        plt.title('Correlation Matrix of abalone features')
        plt.savefig('./abalone_results/abalone_correaltion_map.png')
        plt.clf()
    elif dataset_name == 'CMC':
        #Use catplot&countplot to report the distribution of class 
        plt.figure(figsize = (10, 10))
        sns.countplot(data = data_in, x = 'contraceptive_method')
        plt.title('cmc_class_count')
        plt.savefig('./cmc_results/cmc_class_count.png')
        plt.clf()
        #Use pairplot to report the distribution of features in pairs
        data_features = data_in.drop(columns = ['contraceptive_method'])
        plt.figure(figsize = (30,30))
        sns.pairplot(data = data_features, kind = 'scatter', diag_kind = 'kde')
        plt.title('cmc_features_distribution')
        plt.savefig('./cmc_results/cmc_features_distribution.png')
        plt.clf()
        #Correlation heatmap
        corr = data_features.corr(numeric_only = True)
        plt.figure(figsize = (10,10))
        sns.heatmap(corr, vmin = -1, vmax = 1, annot = True, cmap = 'coolwarm')
        plt.title('Correlation Matrix of cmc features')
        plt.savefig('./cmc_results/cmc_correaltion_map.png')
        plt.clf()
    return 0

def data_preprocessing(dataset_name, data_in, columns, normalize = True):
    x = None
    y = None
    if dataset_name == 'abalone':
        #Reset the M, F, I with one hot encoding method
        data_in = pd.get_dummies(data_in, columns = ['Sex'])
        y = data_in['category']
        x = data_in.drop(columns = ['Rings', 'category'])
        #Data normalization
        if normalize == True:
            min_max_scaler = MinMaxScaler()
            x = min_max_scaler.fit_transform(x)
            x = pd.DataFrame(x, columns = data_in.drop(columns = ['Rings', 'category']).columns)
    elif dataset_name == 'CMC':
        data_in = pd.get_dummies(data_in, columns = ['wife_edu', 'husband_edu', 'husband_occupation', 'standard_of_living_index'])
        y = data_in['contraceptive_method']
        x = data_in.drop(columns = ['contraceptive_method'])
        #Data normalization
        if normalize == True:
            min_max_scaler = MinMaxScaler()
            x = min_max_scaler.fit_transform(x)
            x = pd.DataFrame(x, columns = data_in.drop(columns = ['contraceptive_method']).columns)
    return data_in, x, y

def Decision_Tree(data_in, x, y, max_exp): 
    #Different max_depth choices
    max_depths = np.append(np.arange(3,10), None)
    #Store and report the result
    results1 = []
    for max_depth in max_depths:
        dt_clf_accs = []
        dt_clf_f1s = []
        for i in range(max_exp):
            random_state = i + 2
            #Split the data with SMOTE method
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = random_state)
            smote = SMOTE(random_state = random_state)
            x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
            #Build and train the model
            dt_clf_model = DecisionTreeClassifier(random_state = random_state, max_depth = max_depth)
            dt_clf_model.fit(x_train_smote, y_train_smote)
            #Model effect metrics
            y_pred = dt_clf_model.predict(x_test)
            dt_clf_acc = accuracy_score(y_test, y_pred)
            dt_clf_f1 = f1_score(y_test, y_pred, average = 'weighted')
            dt_clf_accs.append(dt_clf_acc)
            dt_clf_f1s.append(dt_clf_f1)
        results1.append({'max_depth': max_depth, 'accuracy': np.mean(dt_clf_accs), 'accuracy_std': np.std(dt_clf_accs), 'f1_score': np.mean(dt_clf_f1s), 'f1_score_std': np.std(dt_clf_f1s)})
    results1_sheet = pd.DataFrame(results1)
    print('Different indexs with different max_depth of the tree:\n', results1_sheet)
    #Plot the f1_score vs max_depth
    plt.plot(results1_sheet['max_depth'], results1_sheet['f1_score'], marker = 'o')
    plt.xlabel('max_depth')
    plt.ylabel('f1_score')
    plt.title('Decision Tree F1 score vs max_depth')
    plt.savefig('./abalone_results/abalone_DecisionTree_f1_maxdepth.png')
    plt.clf()
    return 0

def Post_Prunned_Tree(data_in, x, y, max_exp):
    #According to the results 1 above, the best max_depth should be 5
    max_depth = 5
    random_state = 42
    #Retrain the Decision Tree model
    dt_clf_best = DecisionTreeClassifier(max_depth = max_depth, random_state = random_state)
    dt_clf_best.fit(x, y)
    #Model effect metrics
    y_pred_dt_clf = dt_clf_best.predict(x)
    dt_clf_best_f1 = f1_score(y, y_pred_dt_clf, average = 'weighted')
    #Plot & visualise the Decision Tree
    plt.figure(figsize = (20, 10))
    plot_tree(dt_clf_best, feature_names = x.columns, class_names = ['0', '1', '2', '3'], filled = True)
    plt.title('Decision Tree Visualization')
    plt.savefig('./abalone_results/abalone_decision_tree.png')
    plt.clf()
    tree = export_text(dt_clf_best, feature_names = x.columns)
    print("The best model of the Decision Tree Classification is:\n", tree)
    #Gain cost complexity pruning path and ccp_alphas
    path = dt_clf_best.cost_complexity_pruning_path(x, y)
    ccp_alphas, impurities = path.ccp_alphas[: - 1], path.impurities
    #Split the data with SMOTE method
    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size = 0.4, random_state = random_state)
    smote = SMOTE(random_state = random_state)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    #For all the ccp_alphas, retrain the model
    ppt_clf_models = []
    for ccp_alpha in ccp_alphas:
        ppt_clf_model = DecisionTreeClassifier(random_state = random_state, ccp_alpha = ccp_alpha)
        ppt_clf_model.fit(x_train_smote, y_train_smote)
        ppt_clf_models.append(ppt_clf_model)
    #Calculate the F1 score of each model to find the best choice of ccp_alphas
    ppt_clf_models = ppt_clf_models[: - 1]
    ccp_alphas = ccp_alphas[: - 1]
    ppt_clf_accs = []
    ppt_clf_f1s = []
    for ppt_clf_model in ppt_clf_models:
        y_pred = ppt_clf_model.predict(x_test)
        ppt_clf_acc = accuracy_score(y_test, y_pred)
        ppt_clf_accs.append(ppt_clf_acc)
        ppt_clf_f1 = f1_score(y_test, y_pred, average = 'weighted')
        ppt_clf_f1s.append(ppt_clf_f1)
    best_index = ppt_clf_f1s.index(max(ppt_clf_f1s))
    best_alpha = ccp_alphas[best_index]
    #Retrain the pruned tree
    ppt_clf_best = DecisionTreeClassifier(random_state = random_state, ccp_alpha = best_alpha)
    ppt_clf_best.fit(x, y)
    #Model effect metrics
    y_pred_ppt_clf = ppt_clf_best.predict(x)
    best_ppt_clf_f1 = f1_score(y, y_pred_ppt_clf, average = 'weighted')
    #Plot & visualise the Decision Tree
    plt.figure(figsize = (20, 10))
    plot_tree(ppt_clf_best, feature_names = x.columns, class_names = ['0', '1', '2', '3'], filled = True)
    plt.title('Post-Pruned Tree Visualization')
    plt.savefig('./abalone_results/abalone_PostPruned_tree.png')
    plt.clf()
    tree2 = export_text(ppt_clf_best, feature_names = x.columns)
    print("The best model of the Post-Pruned Tree Classification is:\n", tree2)
    #Plot the f1_score vs alpha
    plt.figure(figsize = (10, 6))
    plt.plot(ccp_alphas, ppt_clf_f1s, marker = 'o', label = 'Train F1 score')
    plt.xlabel('ccp_alpha')
    plt.ylabel('f1_score')
    plt.title('f1_score vs ccp_alpha')
    plt.savefig('./abalone_results/abalone_PrunedTree_f1_ccp_alpha.png')
    plt.clf()
    #Performance Comparison
    print('The Best ccp_alpha:', best_alpha)
    print('Accuracy of the original Tree:', dt_clf_best.score(x, y))
    print('F1 score of the original Tree:', dt_clf_best_f1)
    print('Accuracy of pruned Tree:', ppt_clf_best.score(x, y))
    print('F1 score of pruned Tree:', best_ppt_clf_f1)
    return 0

def Random_Forest(data_in, x, y, max_exp):
    #Random Forest Model Building and comparation
    n_estimators = [10, 50, 100, 150, 200]
    #Store and report the result
    results2 = []
    for n_estimator in n_estimators:
        rf_clf_accs = []
        rf_clf_f1s = []
        for i in range(max_exp):
            random_state = i + 2
            #Split the data with SMOTE method
            x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size = 0.4, random_state = random_state)
            smote = SMOTE(random_state = random_state)
            x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
            #Build and train the model
            rf_clf_model = RandomForestClassifier(random_state = random_state, n_estimators = n_estimator)
            rf_clf_model.fit(x_train_smote, y_train_smote)
            #Model effect metrics
            y_pred = rf_clf_model.predict(x_test)
            rf_clf_acc = accuracy_score(y_test, y_pred)
            rf_clf_f1 = f1_score(y_test, y_pred, average = 'weighted')
            rf_clf_accs.append(rf_clf_acc)
            rf_clf_f1s.append(rf_clf_f1)
        results2.append({'n_estimator': n_estimator, 'accuracy': np.mean(rf_clf_accs), 'accuracy_std': np.std(rf_clf_accs), 'f1_score': np.mean(rf_clf_f1s), 'f1_score_std': np.std(rf_clf_f1s)})
    results2_sheet = pd.DataFrame(results2)
    print('Different accuracy scores & F1 scores with different max depth of the tree:\n', results2_sheet)
    #Plot the accuracy vs trees number
    plt.plot(results2_sheet['n_estimator'], results2_sheet['f1_score'], marker = 'o')
    plt.xlabel('Number of Trees')
    plt.ylabel('f1_score')
    plt.title('Random Forest F1 score vs Number of Trees')
    plt.savefig('./abalone_results/abalone_RandomForest_f1_n_estimator.png')
    plt.clf()
    return 0

def Boosting(data_in, x, y, max_exp):
    #Here we choose the best n_estimator value above
    n_estimator = 100
    #Comparing with Gradient Boosting and XGBoost
    gb_clf_accs = []
    gb_clf_f1s = []
    xgb_clf_accs = []
    xgb_clf_f1s = []
    for i in range(max_exp):
        random_state = i + 2
        #Split the data with SMOTE method
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = random_state)
        smote = SMOTE(random_state = random_state)
        x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
        #Build and train the Gradient Boosting model
        gb_clf_model = GradientBoostingClassifier(n_estimators = n_estimator, random_state = random_state)
        gb_clf_model.fit(x_train_smote, y_train_smote)
        #Model effect metrics
        y_pred = gb_clf_model.predict(x_test)
        gb_clf_acc = accuracy_score(y_test, y_pred)
        gb_clf_f1 = f1_score(y_test, y_pred, average = 'weighted')
        gb_clf_accs.append(gb_clf_acc)
        gb_clf_f1s.append(gb_clf_f1)
        #Build and train the XGBoost model   
        xgb_clf_model = XGBClassifier(n_estimators = n_estimator, random_state = random_state, use_label_encoder = False, eval_metric = 'mlogloss')
        xgb_clf_model.fit(x_train_smote, y_train_smote)
        #Model effect metrics
        y_pred = xgb_clf_model.predict(x_test)
        xgb_clf_acc = accuracy_score(y_test, y_pred)
        xgb_clf_f1 = f1_score(y_test, y_pred, average = 'weighted')
        xgb_clf_accs.append(xgb_clf_acc)
        xgb_clf_f1s.append(xgb_clf_f1)
    print("Mean of Gradient Boosting Accuracy:", np.mean(gb_clf_accs))
    print("Std of Gradient Boosting Accuracy:", np.std(gb_clf_accs))
    print("Mean of Gradient Boosting F1 Score:", np.mean(gb_clf_f1s))
    print("Std of Gradient Boosting F1 Score:", np.std(gb_clf_f1s))
    print("Mean of XGBoost Accuracy:", np.mean(xgb_clf_accs))
    print("Std of XGBoost Accuracy:", np.std(xgb_clf_accs))
    print("Mean of XGBoost F1 Score:", np.mean(xgb_clf_f1s))
    print("Std of XGBoost F1 score:", np.std(xgb_clf_f1s))
    return 0

def mlp_adam_sgd(data_in, x, y, max_exp):
    #Building a simple Neural Network
    mlp_adam_accs = []
    mlp_adam_f1s = []
    mlp_sgd_accs = []
    mlp_sgd_f1s = []
    for i in range(max_exp):
        random_state = i + 2
        #Split the data with SMOTE method
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = random_state)
        smote = SMOTE(random_state = random_state)
        x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
        #Using Adam optimizer
        mlp_adam_model = MLPClassifier(hidden_layer_sizes = (50, ), solver = 'adam', max_iter = 2000, random_state = random_state)
        mlp_adam_model.fit(x_train_smote, y_train_smote)
        #Model effect metrics
        y_pred = mlp_adam_model.predict(x_test)
        mlp_adam_acc = accuracy_score(y_test, y_pred)
        mlp_adam_f1 = f1_score(y_test, y_pred, average = 'weighted')
        mlp_adam_accs.append(mlp_adam_acc)
        mlp_adam_f1s.append(mlp_adam_f1)
        #Using SGD optimizer  
        mlp_sgd_model = MLPClassifier(hidden_layer_sizes = (50, ), solver = 'sgd', max_iter = 2000, random_state = random_state)
        mlp_sgd_model.fit(x_train_smote, y_train_smote)
        #Model effect metrics
        y_pred = mlp_sgd_model.predict(x_test)
        mlp_sgd_acc = accuracy_score(y_test, y_pred)
        mlp_sgd_f1 = f1_score(y_test, y_pred, average = 'weighted')
        mlp_sgd_accs.append(mlp_sgd_acc)
        mlp_sgd_f1s.append(mlp_sgd_f1)
    print("Mean of Adam Neural Network Accuracy:", np.mean(mlp_adam_accs))
    print("Std of Adam Neural Network Accuracy:", np.std(mlp_adam_accs))
    print("Mean of Adam Neural Network F1 Score:", np.mean(mlp_adam_f1s))
    print("Std of Adam Neural Network F1 Score:", np.std(mlp_adam_f1s))    
    print("Mean of SGD Neural Network Accuracy:", np.mean(mlp_sgd_accs))
    print("Std of SGD Neural Network Accuracy:", np.std(mlp_sgd_accs))
    print("Mean of SGD Neural Network F1 Score:", np.mean(mlp_sgd_f1s))
    print("Std of SGD Neural Network F1 Score:", np.std(mlp_sgd_f1s))
    return 0

def adam_neural_network(data_in, x, y, max_exp):
    #Experiment with different hyperparameters using Adams
    hyperparameters = {'alpha': [0.001, 0.01, 0.1], 'dropout_rate': [0.3, 0.5, 0.7]}
    combinations = product(hyperparameters['alpha'], hyperparameters['dropout_rate'])
    #Store and report the result
    results3 = []
    random.seed(42)
    np.random.seed(42)
    tensorflow.random.set_seed(42)
    for alpha, dropout_rate in combinations:
        #Build the model
        hidden = 50
        output = 4
        learn_rate = 0.01
        nn_adam_accs = []
        nn_adam_f1s = []
        nn_adam = Sequential()
        nn_adam.add(Input(shape = (x.shape[1], )))
        nn_adam.add(Dense(hidden, activation = 'relu', kernel_regularizer = regularizers.l2(alpha)))
        nn_adam.add(Dropout(dropout_rate))
        nn_adam.add(Dense(hidden, activation = 'relu', kernel_regularizer = regularizers.l2(alpha)))
        nn_adam.add(Dropout(dropout_rate))
        nn_adam.add(Dense(output, activation = 'softmax'))
        optimizer = Adam(learning_rate = learn_rate)
        nn_adam.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        for i in range(max_exp):
            #Split the data with SMOTE method
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = i)
            smote = SMOTE(random_state = i)
            x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
            y_train_smote = y_train_smote.astype(int)
            y_test = y_test.astype(int)
            #Train the model
            nn_adam.fit(x_train_smote, y_train_smote, epochs = 20, batch_size = 32, verbose = 0)
            #Model effect metrics
            y_pred = nn_adam.predict(x_test)
            y_pred_clf = y_pred.argmax(axis = 1)
            nn_adam_acc = accuracy_score(y_test, y_pred_clf)
            nn_adam_f1 = f1_score(y_test, y_pred_clf, average = 'weighted')
            nn_adam_accs.append(nn_adam_acc)
            nn_adam_f1s.append(nn_adam_f1)
        results3.append({'alpha': alpha, 'dropout_rate': dropout_rate, 'accuracy': np.mean(nn_adam_accs), 'accuracy_std': np.std(nn_adam_accs), 'f1_score': np.mean(nn_adam_f1s), 'f1_score_std': np.std(nn_adam_accs)})
    results3_sheet = pd.DataFrame(results3)
    print('Different accuracy scores & F1 scores with different hyperparameters:\n', results3_sheet)
    return 0

def cmc_dt_model_building(data_in, x, y, max_exp, max_depth, ccp_alpha):
    dt_clf_model_accs = []
    dt_clf_model_f1s = []
    dt_clf_model_rass = []
    for i in range(max_exp):
        random_state = i + 2
        #Split the data with SMOTE method
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = random_state)
        #Build the Decision Tree model
        dt_clf_model = DecisionTreeClassifier(random_state = random_state, max_depth = max_depth, ccp_alpha = ccp_alpha)
        dt_clf_model.fit(x_train, y_train)
        y_pred_dt_clf = dt_clf_model.predict(x_test)
        #Report the results with the most appropriate metrics
        dt_clf_model_acc = accuracy_score(y_test, y_pred_dt_clf)
        dt_clf_model_f1 = f1_score(y_test, y_pred_dt_clf, average = 'weighted')
        y_prob = dt_clf_model.predict_proba(x_test)
        dt_clf_model_ras = roc_auc_score(y_test, y_prob, multi_class = 'ovr')
        dt_clf_model_accs.append(dt_clf_model_acc)
        dt_clf_model_f1s.append(dt_clf_model_f1)
        dt_clf_model_rass.append(dt_clf_model_ras)
    print('The accuracy score of the Decision Tree:', np.mean(dt_clf_model_accs))
    print('The F1 score of the Decision Tree:', np.mean(dt_clf_model_f1s))
    print('The ROC_AUC score of the Decision Tree:', np.mean(dt_clf_model_rass))
    return 0 

def XGB_clf_model_building(data_in, x, y, max_exp, n_estimator):
    xgb_clf_model_accs = []
    xgb_clf_model_f1s = []
    xgb_clf_model_rass = []
    y = y - 1
    for i in range(max_exp):
        random_state = i + 2
        #Split the data with SMOTE method
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = random_state)
        #Build the Decision Tree model
        xgb_clf_model = XGBClassifier(n_estimators = n_estimator, random_state = random_state, use_label_encoder = False, eval_metric = 'mlogloss')
        xgb_clf_model.fit(x_train, y_train)
        y_pred_xgb_clf = xgb_clf_model.predict(x_test)
        #Report the results with the most appropriate metrics
        xgb_clf_model_acc = accuracy_score(y_test, y_pred_xgb_clf)
        xgb_clf_model_f1 = f1_score(y_test, y_pred_xgb_clf, average = 'weighted')
        y_prob = xgb_clf_model.predict_proba(x_test)
        xgb_clf_model_ras = roc_auc_score(y_test, y_prob, multi_class = 'ovr')
        xgb_clf_model_accs.append(xgb_clf_model_acc)
        xgb_clf_model_f1s.append(xgb_clf_model_f1)
        xgb_clf_model_rass.append(xgb_clf_model_ras)
    print('The accuracy score of the XGBoost:', np.mean(xgb_clf_model_accs))
    print('The F1 score of the XGBoost:', np.mean(xgb_clf_model_f1s))
    print('The ROC_AUC score of the XGBoost:', np.mean(xgb_clf_model_rass))
    return 0 

def main():
    #To avoid the 'Timewall' or 'killed' problem, we set the string variable 'task'
    task = 'VS'#To solve the problem about data visualisation in Part A1
    #task = 'DT'#To solve the problem about Decision Tree in Part A2 and A3
    #task = 'RF'#To solve the problem about Random Forest in Part A4
    #task = 'GB'#To solve the problem about Gradient Boosting and XGBoost in Part A5
    #task = 'SNN'#To solve the problem about Simple Neural Network in Part A6
    #task = 'Adam'#To solve the problem about Adam in Part A7
    #task = 'ND'#To solve the problem about model prediction with new dataset in Part B
    #Part A
    print("{:=^80s}".format("Part A Model Building"))
    dataset_name = 'abalone'
    max_exp = 5
    #Import data
    data, columns = load_data(dataset_name)
    data_in, x, y = data_preprocessing(dataset_name, data, columns, normalize = True)
    if task == 'VS':
        #Part A1
        print("{:=^80s}".format("Part A1 Data visualization"))
        data_visualisation(dataset_name, data, columns)
        print('Finished')
    #Preprocessing
    elif task == 'DT':
        #Part A2
        print("{:=^80s}".format("Part A2 Decision Tree"))
        Decision_Tree(data_in, x, y, max_exp)
        #Part A3
        print("{:=^80s}".format("Part A3 Post Prunned Tree"))
        Post_Prunned_Tree(data_in, x, y, max_exp)
    elif task == 'RF':
        #Part A4
        print("{:=^80s}".format("Part A4 Random Forest"))
        Random_Forest(data_in, x, y, max_exp)
    elif task == 'GB':
        #Part A5
        print("{:=^80s}".format("Part A5 Gradient Boosting and XGBoost"))
        Boosting(data_in, x, y, max_exp)
    elif task == 'SNN':
        #Part A6
        print("{:=^80s}".format("Part A6 Simple Neural Network"))
        mlp_adam_sgd(data_in, x, y, max_exp)
    elif task == 'Adam':
        #Part A7
        print("{:=^80s}".format("Part A7 Adam"))
        adam_neural_network(data_in, x, y, max_exp)
    elif task == 'ND':
        #Part B
        print("{:=^80s}".format("Part B Best model usage towards new dataset CMC"))
        #Import data
        dataset_name = 'CMC'
        data, columns = load_data(dataset_name)
        #Data visualization
        print("{:=^80s}".format("Part B1 Data visualization"))
        data_visualisation(dataset_name, data, columns)
        print('Finished')
        #Apply the best two models above to the new dataset 
        print("{:=^80s}".format("Part B2 Decision Tree Model"))
        max_depth = 5
        ccp_alpha = 0.0021203900958192134
        data_in, x, y = data_preprocessing(dataset_name, data, columns, normalize = True)
        cmc_dt_model_building(data_in, x, y, max_exp, max_depth, ccp_alpha)
        print("{:=^80s}".format("Part B3 XGBoost Model"))
        n_estimator = 100
        XGB_clf_model_building(data_in, x, y, max_exp, n_estimator)
        
    return 0

if __name__ == '__main__':
    main()

