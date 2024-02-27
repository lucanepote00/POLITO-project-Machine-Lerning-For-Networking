# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:15:13 2023

@author: Luca Nepote, Andrea Cuzzi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
import seaborn as sns
import matplotlib.colors as mcolors
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Classification
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


# Clustering
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import (homogeneity_score, silhouette_score, adjusted_rand_score)
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from matplotlib import cm

#Dimenionality Reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

# %%  FUNCTIONS 

# Data Characterization
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns')  # drop columns with NaN
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(
            f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth),
               dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=9)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=9)
    #plt.xticks([])
    #plt.yticks([])
    
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat,  shrink=0.5)
    plt.title(f'Correlation Matrix for {filename}', fontsize=30)
    plt.show()
    
#Classification  
def confusion_matrix(y_true, y_pred):
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    df = pd.DataFrame([x for x in zip(y_true, y_pred)],
                       columns=['y_true', 'y_pred'])
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    df[['samples']] = 1
    confusion = pd.pivot_table(df, index='y_true', 
                               columns='y_pred', 
                               values='samples', 
                               aggfunc=sum)
    return confusion

def GaussianNB_(data_train, data_val, label_train, label_val,title):
    #GAUSSIAN NAIVE BAYES MODEL
    gnb = GaussianNB()
    # Fit a Gaussian Naive Bayes
    gnb.fit(data_train, label_train)
    # Predict the labels for training and validation
    y_train_pred = gnb.predict(data_train)
    y_val_pred = gnb.predict(data_val)
    
    # evaluation
    evaluation(label_train, y_train_pred, label_val, y_val_pred, 'Blues',title)

def KNeighborsClassifier_(data_train, data_val, label_train, label_val, title, k=3):
    knn = KNeighborsClassifier(n_neighbors=k) # dobbiamo scegliere il numero corretto di n_neighbors
    knn.fit(data_train, label_train)
    # Classify both training and validation sets
    y_train_pred = knn.predict(data_train)
    y_val_pred = knn.predict(data_val)
    
    # evaluation
    evaluation(label_train, y_train_pred, label_val, y_val_pred, 'Oranges',title)
    
    
def LDA_Classifier(lda, data_train, data_val, label_train, label_val,title):
    y_train_pred = lda.predict(data_train)
    y_val_pred = lda.predict(data_val) 
    
    # evaluation
    evaluation(label_train, y_train_pred, label_val, y_val_pred, 'Blues',title)
    
def evaluation(y_train,y_train_pred, y_val, y_val_pred, cmap, title):
    # evaluation
    print('\nEvaluation for %s'%title)
    
    print('\nTRAINING CLASSIFICATION REPORT')
    print(classification_report(y_train, y_train_pred))
    print(f"Accuracy : {accuracy_score(y_train, y_train_pred)}")
   
    print('\n\nVALIDATION CLASSIFICATION REPORT')
    print(classification_report(y_val, y_val_pred))
    print(f"Accuracy : {accuracy_score(y_val, y_val_pred)}")
    
    # Training Confusion Matrix
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    # Get the confusion matrix - training
    confusion_train = confusion_matrix(y_train.reshape(len(y_train)), y_train_pred)
    # Normalize by rows
    confusion_train = round(confusion_train.divide(confusion_train.sum(1), axis=0).fillna(0)*100)
    # Visualize
    sns.heatmap(confusion_train, cmap=cmap, annot=True, vmin=0, vmax=100, 
                ax=axs[0], cbar_kws={'label':'Occurrences'})
    axs[0].set_xlabel('Prediction')
    axs[0].set_ylabel('True')
    axs[0].set_title('Training')
    
    # Validation confusion matrix
    confusion_val = confusion_matrix(y_val.reshape(len(y_val)), y_val_pred)
    # Normalize by rows
    confusion_val = round(confusion_val.divide(confusion_val.sum(1), axis=0).fillna(0)*100)
    # Visualize
    sns.heatmap(confusion_val, cmap=cmap, annot=True, vmin=0, vmax=100, 
                ax=axs[1], cbar_kws={'label':'Occurrences'})
    axs[1].set_xlabel('Prediction')
    axs[1].set_ylabel('True')
    axs[1].set_title('Validation')
    
    plt.suptitle(title)
    plt.show()
    
def scatter_plot1(unique_labels,df):
    column = df.columns
    
    plt.figure()
    for lab in unique_labels:
        subdf = df[df.label == lab]
        plt.scatter(subdf[column[0]], subdf[column[1]],
                      s=5, color=color_map[lab], label=lab.capitalize())
    plt.xlabel(column[0])
    plt.ylabel(column[1])
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
def scatter_plot2(axs, unique_labels,df):
    column = df.columns
    for lab in unique_labels:
        subdf = df[df.label == lab]
        axs.scatter(subdf[column[0]], subdf[column[1]],
                      s=5, color=color_map[lab], label=lab.capitalize())
    axs.set_xlabel(column[0])
    axs.set_ylabel(column[1])
    axs.grid()
    
# Dimensionality Reduction LDA
def LinearComponentDistribution(lda, train_data):
    # Get the LDA scores
    lda_scores = pd.DataFrame(lda.coef_, columns=train_data.columns).T
    lda_scores = lda_scores.rename(columns={0: 'LC'})
    # Normalize between 0 and 1
    lda_scores = (lda_scores/lda_scores.abs().max()).sort_values('LC')

    # Plot the distribution of the Linear Component
    plt.figure(figsize=(5, 3.5))
    ax = plt.gca()
    lda_scores.plot.barh(ax=ax, legend=False)
    ax.grid()
    ax.set_xlabel('Loadings')
    ax.set_ylabel('Feature')
    plt.yticks(fontsize=7)
    ax.set_title('LDA')

def DensityDistribution(title,unique_labels,lda_train,lda_val,colors):
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    plt.suptitle(title)
    
    for lab in unique_labels:
        sns.distplot(lda_train[lda_train['label'] == lab].drop(
            ['label'], axis=1), color=colors[lab], label=lab, ax=axs[0])
    axs[0].grid()
    axs[0].set_ylabel('Density')
    axs[0].set_xlabel('Linear Component')
    axs[0].set_title('Training')

    for lab in unique_labels:
        sns.distplot(lda_val[lda_val['label'] == lab].drop(
            ['label'], axis=1), color=colors[lab], label=lab, ax=axs[1])
    axs[1].grid()
    axs[1].set_ylabel('Density')
    axs[1].set_xlabel('Linear Component')
    axs[1].legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].set_title('Validation')

    plt.show()


# Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

        
# TASK 3
def run_DT(X_train_s, y_train,X_val_s,y_val, FLAG_accuracy):

    clf = DecisionTreeClassifier(random_state=15)
    clf.fit(X_train_s, y_train)
    # y_train_pred = clf.predict(X_train_s)
    y_val_pred = clf.predict(X_val_s)

    best_clf = clf
    
    if FLAG_accuracy:
        best_accuracy = accuracy_score(y_val, y_val_pred)
    else:
        best_accuracy = f1_score(y_val, y_val_pred, average='macro')
        

    trend_accuracy = []
    trend_accuracy.append(best_accuracy)
    configs = []
    configs.append("Default")

    for max_depth in range(5,21,5):
        for min_samples_split in range(5,21,5):
            for min_impurity_decrease in np.arange(0.1, 0.61, 0.1):
                configs.append("MD: %d MSS: %d MID: %.2f"%(max_depth,min_samples_split,min_impurity_decrease))
                
                clf = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_impurity_decrease=min_impurity_decrease, random_state=15)

                clf.fit(X_train_s, y_train)
                # Predict the labels for validation
                y_val_pred = clf.predict(X_val_s)


                if FLAG_accuracy:
                    accuracy = accuracy_score(y_val, y_val_pred)
                else:
                    accuracy = f1_score(y_val, y_val_pred, average='macro')
                    

                trend_accuracy.append(accuracy)
                
                if(best_accuracy < accuracy): 
                    best_clf = clf
                    best_accuracy = accuracy
   
    fig, ax = plt.subplots(figsize=(5,4))    
    ax.plot(configs,trend_accuracy)
    selected_labels = [configs[i] for i in range(0,len(configs),5)]
    ax.set_xticks([i for i in range(0,len(configs),5)])
    ax.set_xticklabels(selected_labels, rotation = 90)
    ax.set_title("DT performance")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    plt.close()     
    
    return best_clf

def run_RF(X_train_s, y_train,X_val_s,y_val, FLAG_accuracy):

    clf = RandomForestClassifier(random_state=15)
    clf.fit(X_train_s, y_train)
    y_val_pred = clf.predict(X_val_s)

    best_clf = clf
    if FLAG_accuracy:
        best_accuracy = f1_score(y_val, y_val_pred, average='macro')
    else:
        best_accuracy = accuracy_score(y_val, y_val_pred)
 
    trend_accuracy = []
    trend_accuracy.append(best_accuracy)
    configs = []
    configs.append("Default")
    
    for n_estimators in range(20,101,20):
        for max_depth in range(5,21,5):
            for min_samples_split in range(5,21,5):
                for min_impurity_decrease in np.arange(0.1, 0.61, 0.15):
                    configs.append("NE: %d MD: %d MSS: %d MID: %.2f"%(n_estimators,max_depth,min_samples_split,min_impurity_decrease))
                    
                    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,min_impurity_decrease=min_impurity_decrease, random_state=15)

                    clf.fit(X_train_s, y_train)
                    # Predict the labels for validation
                    y_val_pred = clf.predict(X_val_s)
                    
                    if FLAG_accuracy:
                        accuracy = accuracy_score(y_val, y_val_pred)
                    else:
                        accuracy = f1_score(y_val, y_val_pred, average='macro')
                    trend_accuracy.append(accuracy)
                    
                    if(best_accuracy < accuracy): 
                        best_clf = clf
                        best_accuracy = accuracy
                        
    fig, ax = plt.subplots(figsize=(5,4))    
    ax.plot(configs,trend_accuracy)
    selected_labels = [configs[i] for i in range(0,len(configs),5)]
    ax.set_xticks([i for i in range(0,len(configs),5)])
    ax.set_xticklabels(selected_labels, rotation = 90)
    ax.set_title("RF performance")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    plt.close()     

    return best_clf
            

def get_feature_importance(df,clf):

    X = df.drop(columns=['label']).to_numpy()
    y = df[['label']].to_numpy()
    scaler = StandardScaler()
    scaler.fit(X)
    X_s = scaler.transform(X)

    clf.fit(X_s, y)

    features = list(df.columns)
    features.remove("label")
    feature_importance = {features[i]: clf.feature_importances_[i] for i in range(len(clf.feature_importances_))}

    feature_importance_sorted = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1])}

    return feature_importance_sorted


def compute_recurrent_feature_elimination(clf,df,feature_importance, FLAG_accuracy):

    global_performance_train = []
    global_performance_val = []

    for feature in feature_importance:
        X = df.drop(columns=['label'], axis=1).to_numpy()
        y = df[['label']].to_numpy()

        X_train, X_val, y_train, y_val = train_test_split(X, y,  stratify=y, train_size=.7, random_state=15)
        y_train, y_val = np.ravel(y_train), np.ravel(y_val)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

        clf.fit(X_train_s, y_train)
        y_train_pred = clf.predict(X_train_s)
        y_val_pred = clf.predict(X_val_s)
        
        if FLAG_accuracy:
            accuracy_train = accuracy_score(y_train, y_train_pred)
            accuracy_val   = accuracy_score(y_val, y_val_pred)
        else:
            accuracy_train = f1_score(y_train, y_train_pred, average='macro')
            accuracy_val   = f1_score(y_val, y_val_pred, average='macro')

        global_performance_train.append(accuracy_train)
        global_performance_val.append(accuracy_val)

        #Drop Current Least Important Feature
        df = df.drop(columns=[feature])
    return global_performance_train, global_performance_val


def recursive_stratified_elimination(clf,df,percentage,unique_label, FLAG_accuracy):
    
    global_performance_train = []
    global_performance_val = []
    
    X = df.drop(columns=['label'], axis=1).to_numpy()
    y = df[['label']].to_numpy()

    for value in percentage:
        #usiamo la funzione per togliere i dati
        sub_X, sub_X_dropped, sub_y, sub_y_dropped = train_test_split(X, y,  stratify=y, train_size=1-value/100, random_state=15)
        sub_y, sub_y_dropped = np.ravel(sub_y), np.ravel(sub_y_dropped)
        
        X_train, X_val, y_train, y_val = train_test_split(sub_X, sub_y,  stratify=sub_y, train_size=.7, random_state=15)
        y_train, y_val = np.ravel(y_train), np.ravel(y_val)
            
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)
    
        clf.fit(X_train_s, y_train)
        y_train_pred = clf.predict(X_train_s)
        y_val_pred = clf.predict(X_val_s)
    
        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_val   = accuracy_score(y_val, y_val_pred)
    
        global_performance_train.append(accuracy_train)
        global_performance_val.append(accuracy_val)
        
    return global_performance_train, global_performance_val


# Clustering

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    _contingency_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    a = np.sum(np.amax(_contingency_matrix, axis=0))
    b = np.sum(_contingency_matrix)
    return a/b


# %%  MAIN 

if __name__ == '__main__':
    

    df_train = pd.read_csv('Project2_training.csv')
    df_val = pd.read_csv('Project2_validation.csv')
    
    random.seed(5020)
    
    
    # Dataset Analisys
    data_ana = 1
    if data_ana:
        
        # Dataset features
        columns = df_train.columns
        columns = [str.strip() for str in columns.astype('str')] #to make sure we remove spaces
        df_train.columns = columns
        
        # labels - Domain Name
        unique_labels = df_train.label.unique()
        
        
        # associate a color to each label for representation purpose
        C = list(mcolors.CSS4_COLORS)
        idx = random.sample(range(len(C)), len(unique_labels))
        color = np.take(C, idx)
        color_map = dict()
        color_map = {domain_name:c for domain_name,c in zip(unique_labels,color)}
        
        
        # indexing in time - not account for classification purpose
        df_train.index = pd.DatetimeIndex(df_train.time)
        df_train = df_train.drop(["time"], axis=1)
        
        df_val.index = pd.DatetimeIndex(df_val.time)
        df_val = df_val.drop(["time"], axis=1)

        
        # General values
        print('Distinct client IPs:')
        print(df_train.value_counts('c_ip').shape[0])
        print('Number of different domain names:')
        print(unique_labels.shape[0])
        print('Number of different client ports')
        print(df_train.value_counts('_c_port').shape[0])
        print('Number of different server ports')
        print(df_train.value_counts('_s_port').shape[0]) #è uguale per tutti, https, quindi si può togliere
    
        # Descriptive statistics
        df_train_describe=df_train.describe()
        print(df_train_describe)
    
        # Missing values
        print(df_train.isna().sum().sum()) #non ci sono valori NAN
    
        # Data types
        print(df_train.dtypes)
    
        col = ['c_ip','label','_c_port','_s_port']
        value_data = [df_train.value_counts('c_ip').shape[0], df_train.value_counts('label').shape[0], df_train.value_counts('_c_port').shape[0], df_train.value_counts('_s_port').shape[0]]
        table = plt.table(cellText=[value_data], rowLabels=['# of unique values'], colLabels=['client IPs','domain names','client ports','server ports'], loc='center')
        plt.show()
        
        
        # not-balanced Dataset
        values=dict()
        
        for label in unique_labels:
            
            sub_df_train = df_train[df_train['label']==label]
            values[label]=sub_df_train['label'].count() #not balanced dataset 
            
        
        maximum = max(values.items(), key=operator.itemgetter(1))
        print(maximum)
        minimum = min(values.items(), key=operator.itemgetter(1)) 
        print(minimum) 
        values = sorted(values.items(), key=operator.itemgetter(1), reverse=True)
        
        x = [val[0] for val in values]
        y = [val[1] for val in values]
        plt.bar(x, y)
        plt.xticks(rotation=90)
        plt.ylabel('# of samples per Domain Name')
        plt.title('Samples distribution per label')
        plt.grid()
        plt.show()
   
        
    #  investigating the dataset
      
    df_train.dataframeName = 'Entire Dataset'
    
    plotCorrelationMatrix(df_train, 19)
    plt.show()
    
    df_train['_c_bytes_retx'].value_counts() #we noted that the biggest part of the data is equal to 0
    df_train['_c_mss'].value_counts()
    df_train['_c_mss'].unique().shape[0] # ci sono solo 14 valori diversi
    
    # _c_pkts_all and _c_bytes_all are correlated so they contain the same info(from the correlation matrix)
    
    df_train['_c_pkts_fc'].value_counts() #sono tutti uguali a zero
    df_train['_c_pkts_ooo'].value_counts() #quasi tutti a zero, la maggior parte sono arrivati in ordine 
    df_train['_c_pkts_reor'].value_counts() #quasi tutti zero
    df_train['_c_pkts_unfs'].value_counts() #tutti zeri. Non ci sono FIN senza ACK
    df_train['_c_pkts_unrto'].value_counts()

    df_train['_c_port']=np.array(df_train['_c_port'].values,dtype='int32')
    x = np.arange(0,65536)
   
    plt.figure(figsize=(10,7))
    for lab in unique_labels:
        subdf = df_train[df_train['label'] == lab]
        ECDF_Port = np.array(subdf['_c_port'].value_counts().reset_index())
        y = np.zeros(len(x))
        
        for port,count in ECDF_Port: 
            y[port] = count

        for i in np.arange(len(y)-1):
            y[i+1] += y[i]
            y[i+1] = int(y[i+1])
        
        y = y/np.max(y)

        plt.plot(x,y,label = lab)       
    plt.xlabel('Client Port')
    plt.title('ECDF per classes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()

    plt.show()
    
    #client 
    df_train['_c_rst_cnt'].value_counts() #la maggior parte sono 1, quindi vengono usati per terminare e resettare la connessione in modo brutale 
    df_train['_c_sack_cnt'].value_counts() #la maggior parte sono zeri
    df_train['_c_sack_opt'].value_counts() #tutti 1, tutti quanti supportano il sack, si può togliere 
    df_train['_c_syn_cnt'].value_counts() #quasi tutti uguali a 1
    df_train['_c_syn_retx'].value_counts() #tutti zeri, si possono togliere
    df_train['_c_tm_opt'].value_counts() #abbastanza equamente suddivisi tra 0 e 1 
    df_train['_c_win_0'].value_counts() #la maggior parte sono zeri
    df_train['_c_win_scl'].value_counts() #valore massimo uguale a 8
    
    
    #server
    df_train['_s_appdataB'].value_counts()
    df_train['_s_ack_cnt'].value_counts()
    df_train['_s_bytes_retx'].value_counts() #quasi tutti 0
    df_train['_s_f1323_opt'].value_counts() #quasi tutti 1
    df_train['_s_fin_cnt'].value_counts()
    df_train['_s_mss'].value_counts() #praticamente tutti uguali
    df_train['_s_mss_max'].value_counts() #tanti valori uguali ma varianza comunque elevata 
    df_train['_s_mss_min'].value_counts() #come sopra
    df_train['_s_pkts_all'].value_counts() #utile
    df_train['_s_pkts_data_avg'].value_counts()
    df_train['_s_pkts_dup'].value_counts() #tanti 0, non ci sono pacchetti duplicati
    df_train['_s_pkts_fc'].value_counts() #most are zeros
    df_train['_s_pkts_fs'].value_counts() #most are zeros
    df_train['_s_pkts_ooo'].value_counts() #most are zeros
    df_train['_s_pkts_push'].value_counts() #tanti valori bassi, di valori alti ce ne sono pochi
    df_train['_s_pkts_reor'].value_counts() #most are zeros 
    df_train['_s_pkts_retx'].value_counts() #few retx packets 
    df_train['_s_pkts_rto'].value_counts() #most are zeros
    df_train['_s_pkts_unfs'].value_counts() #all are zeros, not useful 
    df_train['_s_pkts_unk'].value_counts() #most are zeros, bigger number for small values
    df_train['_s_pkts_unrto'].value_counts() #most are zeros 
    df_train['_s_pktsize1'].value_counts()
    df_train['_s_port'].value_counts() #they have all the same port #443
    df_train['_s_rst_cnt'].value_counts() #most are zeros, not too much 1s
    df_train['_s_sack_cnt'].value_counts() #most are zeros
    df_train['_s_sack_opt'].value_counts() #all are ones
    df_train['_s_sit1'].value_counts() 
    df_train['_s_syn_cnt'].value_counts() # most are ones
    df_train['_s_syn_retx'].value_counts()  #they are all zeros 
    df_train['_s_tm_opt'].value_counts() #half ones and half zeros 
    df_train['_s_ttl_max'].value_counts()
    df_train['_s_ttl_min'].value_counts() #they are almost equal 
    #_s_win_0 are all zeros 
    df_train['_s_win_scl'].value_counts() #dovrebbero essere 0 e 1, ma c'è 9 al posto di 1.
    #tutti quelli con 0 e 1 devono essere trattati come categorici 
    df_train['_tls_session_stat'].value_counts() #values 1, 0, 3
    
    #plot of the duration
    sns.displot(data=df_train[['_durat']],kind='hist', x='_durat', kde=True)
    plt.grid()
    plt.show() 

    c=df_train[['_tls_session_stat','label']]
    scatter_plot1(unique_labels,c) 
    
    
    #trattiamo come categoriche le features '_c_ip', '_c_port', '_s_port' e '_tls_session_stat' 
    
    df_train['_c_port'] = df_train['_c_port'].apply(str) #transformed in str
    df_train['_c_port'] = df_train['_tls_session_stat'].apply(str) 
    df_train['_c_port'] = df_train['_s_port'].apply(str)
    
    new_dataset = pd.get_dummies(df_train.drop(['label'], axis=1))
    
    
    # analisys per label

    # prova - '_c_pkts_ooo','_c_pkts_reor', _c_cwin_min, _c_cwin_max, _c_win_max, _c_win_min
    fig, ax = plt.subplots(3,figsize=(10,15))
    
    scatter_plot2(ax[0], unique_labels,df_train[['_c_pkts_ooo','_c_pkts_reor','label']])
    scatter_plot2(ax[1], unique_labels,df_train[['_c_cwin_min','_c_win_min','label']])
    scatter_plot2(ax[2], unique_labels,df_train[['_c_cwin_max','_c_win_max','label']])
    
    plt.legend(bbox_to_anchor=(1.05, 2.1), loc='upper left')
    plt.show()
    
    # max-min / std
    
    # _c_rtt_max, _c_rtt_min, _c_rtt_std, _c_rtt_avg
    
    c = pd.DataFrame({'Standardized rtt': (df_train['_c_rtt_max'].values - df_train['_c_rtt_min'].values)/df_train['_c_rtt_std'], 
                     'rtt avg Mean': df_train['_c_rtt_avg'], 
                     'label': df_train['label'].values}).fillna(0)
  
    scatter_plot1(unique_labels,c)
    
    # _s_rtt_max, _s_rtt_min, _s_rtt_std, _s_rtt_avg
    
    c = pd.DataFrame({'Standardized rtt': (df_train['_s_rtt_max'].values - df_train['_s_rtt_min'].values)/df_train['_s_rtt_std'], 
                     'rtt avg Mean': df_train['_s_rtt_avg'], 
                     'label': df_train['label'].values}).fillna(0)

    scatter_plot1(unique_labels,c)
    

    #############################################################
    
    not_useful_features = []
    
    for col in df_train.columns:
        vec = df_train[col].value_counts()
        if 0 in vec.index and vec[0] > 2 * maximum[1]:
            not_useful_features.append(col)
    
    # values distribution of the 'not_useful_features'
    for feature in not_useful_features:
        print(feature, df_train[feature].value_counts())

    # correlated fearures --> the identical columns are considered
    corr_matrix_clear = df_train.corr().abs()

    upper = corr_matrix_clear.where(
        np.triu(np.ones(corr_matrix_clear.shape), k=1).astype(np.bool))
    # Find index of columns with correlation greater than 0.95 - eliminate also the equal columns
    high_correlated_features = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    new_dataset = df_train.drop(high_correlated_features,axis = 1)
    new_dataset.dataframeName = 'Uncorrelated Dataset'
    plotCorrelationMatrix(new_dataset, 19)
    plt.show()
    
    
    useful_feature = set(df_train.columns) - set(not_useful_features) - set(high_correlated_features)
    useful_feature -= set(['_c_port', '_s_port','c_ip'])
    print('# of useful features: '+str(len(useful_feature)))
    print('# of high correlated features: '+str(len(high_correlated_features)))
    print('# of not useful features: '+str(len(not_useful_features)))
    print('total # of features: '+str(len(df_train.columns)))
    
        
    # eliminate categorical data
    df_train = df_train.drop(['_c_port', '_s_port','c_ip', '_tls_session_stat'],axis=1)
    
    # %% TASK 2 - common with task 3
    
    #X = new_dataset.to_numpy()
    X = df_train.drop(['label'],axis=1)
    y = df_train[['label']].to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=.7, random_state=15)
    
    # standardized dataset
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)
    
    
    # %% 2.1 SUPERVISED CLASSIFICATION with the original features
     
    #we study Gaussian Naive-Bayes, KNN, LDA, Decision Tree, Random Forest 
    
    #GAUSSIAN NAIVE BAYES MODEL 
    GaussianNB_(data_train=X_train, data_val=X_val, label_train=y_train, label_val=y_val, 
                title='GAUSSIAN NAIVE BAYES MODEL WITHOUT PCA AND LDA')
    
    #GAUSSIAN NAIVE BAYES MODEL with standardized features
    GaussianNB_(data_train=X_train_s, data_val=X_val_s, label_train=y_train, label_val=y_val, 
                title='GAUSSIAN NAIVE BAYES MODEL WITHOUT PCA AND LDA')
       
    #3-NN classifier
    KNeighborsClassifier_(data_train=X_train, data_val=X_val, label_train=y_train, label_val=y_val, 
                title='3-NN classifier WITHOUT PCA AND LDA',k=3)
    
    # K-NN - Search of best hyper-parameter (K)
    search_K = 0
    if search_K == 1:
        # best hyper-parameter k   
        Score_KNN = pd.DataFrame(columns = ['k_neighbours','training_f1_score', 'validation_f1_score'])
        
        
        for k in range(2, 20):
            # Fit a knn
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_s, y_train)
            # Classify both training and validation sets
            y_train_pred = knn.predict(X_train_s)
            y_val_pred = knn.predict(X_val_s)
            # Get the training report
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            train_result = train_report['macro avg']['f1-score']
            # Get the validation report
            val_report = classification_report(y_val, y_val_pred, output_dict=True)
            val_result = val_report['macro avg']['f1-score']
            
            new_row = {'k_neighbours':k,'training_f1_score':train_result, 'validation_f1_score':val_result}
            
            Score_KNN = Score_KNN.append(new_row,ignore_index=True)

        # Get k leading to the highest f1_score - Training
        Score_KNN['training_f1_score'] = Score_KNN['training_f1_score'].astype(float)
        index = Score_KNN['training_f1_score'].argmax()
        
        best_K_train, best_val_train = Score_KNN.iloc[index][['k_neighbours', 'training_f1_score']].values
        best_K_train = int(best_K_train)
        
        # Get k leading to the highest f1_score - validation
        Score_KNN['validation_f1_score'] = Score_KNN['validation_f1_score'].astype(float)
        index = Score_KNN['validation_f1_score'].argmax()
        
        best_K_val, best_val_val = Score_KNN.iloc[index][['k_neighbours', 'validation_f1_score']].values
        best_K_val = int(best_K_val)
        
        #Plot
        plt.figure(figsize=(5, 3.5))
        plt.plot(Score_KNN['k_neighbours'], Score_KNN['training_f1_score'], label='Training', color='r')
        plt.scatter(best_K_train,best_val_train, marker='o',s=50, c='red')
        plt.plot(Score_KNN['k_neighbours'], Score_KNN['validation_f1_score'], color='b')
        plt.scatter(best_K_val,best_val_val, marker='o',s=50, c='blue')
        plt.grid()
        plt.xlabel('k-Nearest-Neighborhood radius')
        plt.ylabel('Macro avg. F1-Score')
        plt.xlim(1, 21)
        plt.xticks(np.arange(2,20))
        plt.legend()
        plt.show()
        
    else:
        best_K_val = 3
    
    #3-NN classifier with standardized features
    KNeighborsClassifier_(data_train=X_train_s, data_val=X_val_s, label_train=y_train, label_val=y_val, 
                title='3-NN classifier WITHOUT PCA AND LDA',k=best_K_val)
    
    
    # Decision Tree 
    Score_DT_init = pd.DataFrame(columns=['max_depth', 'min_samples_split', 'min_impurity_decrease', 'Accuracy', 'F1_Score'])
    
    for max_depth in range(5,21,5):
        for min_samples_split in range(5,21,5):
            for min_impurity_decrease in np.arange(0.1, 0.61, 0.1):

                tree = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_impurity_decrease=min_impurity_decrease, random_state=15)
                tree.fit(X_train_s,y_train)
                
                Acuracy_score = cross_val_score(tree, X_val_s, y_val, cv=10, scoring='accuracy')
                F1_score = cross_val_score(tree, X_val_s, y_val, cv=10, scoring='f1_macro') 
                
                new_row = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_impurity_decrease':min_impurity_decrease, 'Accuracy':np.mean(Acuracy_score), 'F1_Score':np.mean(F1_score)}
                Score_DT_init = Score_DT_init.append(new_row, ignore_index=True)
    
    Score_DT_init['F1_Score'] = Score_DT_init['F1_Score'].astype(float)
    index = Score_DT_init['F1_Score'].argmax()
    
    BEST_DT_max_depth_init, BEST_DT_min_samples_split_init, BEST_DT_min_impurity_decrease_init = Score_DT_init.iloc[index][['max_depth', 'min_samples_split', 'min_impurity_decrease']].values
    BEST_DT_min_samples_split_init = int(BEST_DT_min_samples_split_init)
    BEST_DT_max_depth_init = int(BEST_DT_max_depth_init)
    
    
    # Random Forest 
    Score_RF_init = pd.DataFrame(columns=['n_estimators','max_depth', 'min_samples_split', 'min_impurity_decrease', 'Accuracy', 'F1_Score'])
    
    for n_estimators in range(20,101,20):
        for max_depth in range(5,21,5):
            for min_samples_split in range(5,21,5):
                for min_impurity_decrease in np.arange(0.1, 0.61, 0.15):
                    
                    forest = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,min_impurity_decrease=min_impurity_decrease, random_state=15)
                    forest.fit(X_train_s,y_train)
                    
                    Acuracy_score = cross_val_score(forest, X_val_s, y_val, cv=10, scoring='accuracy')
                    F1_score = cross_val_score(forest, X_val_s, y_val, cv=10, scoring='f1_macro') 
                    
                    new_row = {'n_estimators': n_estimators,'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_impurity_decrease':min_impurity_decrease, 'Accuracy':np.mean(Acuracy_score), 'F1_Score':np.mean(F1_score)}
                    Score_RF_init = Score_RF_init.append(new_row, ignore_index=True)
    
    Score_RF_init['F1_Score'] = Score_RF_init['F1_Score'].astype(float)
    index = Score_RF_init['F1_Score'].argmax()
    
    BEST_RF_max_depth_init, BEST_RF_min_samples_split_init, BEST_RF_min_impurity_decrease_init, BEST_RF_n_estimators_init = Score_RF_init.iloc[index][['max_depth', 'min_samples_split', 'min_impurity_decrease','n_estimators']].values
    BEST_RF_min_samples_split_init = int(BEST_RF_min_samples_split_init)
    BEST_RF_max_depth_init = int(BEST_RF_max_depth_init)
    BEST_RF_n_estimators_init = int(BEST_RF_n_estimators_init)
    
    
    
    ## Different features
    
    # Remove the unwated features seen in the first task
    features_to_eliminate = not_useful_features + high_correlated_features
    X = X.drop(np.unique(features_to_eliminate),axis=1)
    
    # create new features
    ADD_Feature = 1
    if ADD_Feature == 1:
        #X = new_dataset.to_numpy()
        X = df_train.drop(['label'],axis=1)
        y = df_train[['label']].to_numpy()
        
        # ampiezza della finestra di congestione
        X['_amplitude_c_cwin'] = X['_c_cwin_max'] - X['_c_cwin_min']
        # densita dei dati trasmessi rispetto alla dimensione media dei pacchetti
        X['_data_density'] = X['_c_pkts_data_avg']/X['_c_msgsize_count']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=.7, random_state=15)
    
    # standardized dataset
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)
    
 
    # Dimensionality reduction
    ## LDA ##
    lda_col = ['LDA'+str(N) for N in range(len(unique_labels)-1)]

    lda= LDA()
    lda.fit(X_train_s,y_train)
    
    
    # Transform the training samples
    lda_train = lda.transform(X_train_s)
    lda_train = pd.DataFrame(lda_train, columns=lda_col)
    lda_train['label'] = y_train
    
    # Transform the validation samples
    lda_val = lda.transform(X_val_s)
    lda_val = pd.DataFrame(lda_val, columns=lda_col)
    lda_val['label'] = y_val
   
    LinearComponentDistribution(lda, X)
    DensityDistribution('LDA', unique_labels, lda_train, lda_val, color_map)


    top_importance = np.argmax(lda.coef_, axis=1)
    top_features = df_train.drop(['c_ip','label'],axis=1).columns[top_importance]
    for x in zip(lda.classes_, top_features):
        print(f'{x[0]} -> {x[1]}')
    
    
    # classification after dimensionality reduction and features ingeneering
    #GAUSSIAN NAIVE BAYES MODEL WITH LDA
    GaussianNB_(data_train=lda_train.drop(['label'], axis=1).to_numpy(), data_val=lda_val.drop(['label'], axis=1).to_numpy(),
                label_train=lda_train['label'].to_numpy(), label_val=lda_val['label'].to_numpy(),
                title='GAUSSIAN NAIVE BAYES MODEL WITH LDA ')
        
    #3-NN classifier WITH LDA
    KNeighborsClassifier_(data_train=lda_train.drop(['label'], axis=1).to_numpy(), data_val=lda_val.drop(['label'], axis=1).to_numpy(),
                label_train=lda_train['label'].to_numpy(), label_val=lda_val['label'].to_numpy(),
                title='3-NN classifier WITH LDA ',k=best_K_val)
    
    #LDA classifier as Tied_Covariance
    LDA_Classifier(lda = lda, data_train=X_train_s, data_val=X_val_s,
                label_train=lda_train['label'].to_numpy(), label_val=lda_val['label'].to_numpy(),
                title='LDA classifier')
    
    # Decision Tree 
    Score_DT = pd.DataFrame(columns=['max_depth', 'min_samples_split', 'min_impurity_decrease', 'Accuracy', 'F1_Score'])
    
    for max_depth in range(5,21,5):
        for min_samples_split in range(5,21,5):
            for min_impurity_decrease in np.arange(0.1, 0.61, 0.1):
                
                tree = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_impurity_decrease=min_impurity_decrease, random_state=15)
                tree.fit(lda_train.drop(['label'],axis=1),lda_train['label'])
                
                Acuracy_score = cross_val_score(tree, lda_val.drop(['label'], axis=1), lda_val['label'], cv=10, scoring='accuracy')
                F1_score = cross_val_score(tree, lda_val.drop(['label'], axis=1), lda_val['label'], cv=10, scoring='f1_macro') 
                
                new_row = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_impurity_decrease':min_impurity_decrease, 'Accuracy':np.mean(Acuracy_score), 'F1_Score':np.mean(F1_score)}
                Score_DT = Score_DT.append(new_row, ignore_index=True)
    
    Score_DT['F1_Score'] = Score_DT['F1_Score'].astype(float)
    index = Score_DT['F1_Score'].argmax()
    
    BEST_DT_max_depth, BEST_DT_min_samples_split, BEST_DT_min_impurity_decrease = Score_DT.iloc[index][['max_depth', 'min_samples_split', 'min_impurity_decrease']].values
    BEST_DT_min_samples_split = int(BEST_DT_min_samples_split)
    BEST_DT_max_depth = int(BEST_DT_max_depth)
    
    
    # Random Forest 
    Score_RF = pd.DataFrame(columns=['n_estimators','max_depth', 'min_samples_split', 'min_impurity_decrease', 'Accuracy', 'F1_Score'])
    
    for n_estimators in range(20,101,20):
        for max_depth in range(5,21,5):
            for min_samples_split in range(5,21,5):
                for min_impurity_decrease in np.arange(0.1, 0.61, 0.15):
                    
                    forest = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,min_impurity_decrease=min_impurity_decrease, random_state=15)
                    forest.fit(lda_train.drop(['label'],axis=1), lda_train['label'])
                    
                    Acuracy_score = cross_val_score(forest, lda_val.drop(['label'], axis=1), lda_val['label'], cv=10, scoring='accuracy')
                    F1_score = cross_val_score(forest, lda_val.drop(['label'], axis=1), lda_val['label'], cv=10, scoring='f1_macro') 
                    
                    new_row = {'n_estimators': n_estimators,'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_impurity_decrease':min_impurity_decrease, 'Accuracy':np.mean(Acuracy_score), 'F1_Score':np.mean(F1_score)}
                    Score_RF = Score_RF.append(new_row, ignore_index=True)
    
    Score_RF['F1_Score'] = Score_RF['F1_Score'].astype(float)
    index = Score_RF['F1_Score'].argmax()
    
    BEST_RF_max_depth, BEST_RF_min_samples_split, BEST_RF_min_impurity_decrease, BEST_RF_n_estimators = Score_RF.iloc[index][['max_depth', 'min_samples_split', 'min_impurity_decrease','n_estimators']].values
    BEST_RF_min_samples_split = int(BEST_RF_min_samples_split)
    BEST_RF_max_depth = int(BEST_RF_max_depth)
    BEST_RF_n_estimators = int(BEST_RF_n_estimators)
    
    ## Final evaluation
    
    ## training
    # Remove the categorical data
    df_classification_train = df_train
    # Remove the unwated features seen in the first task
    df_classification_train = df_classification_train.drop(np.unique(features_to_eliminate),axis=1)
    
    # add the new features
    ADD_Feature = 1
    if ADD_Feature == 1:
        # ampiezza della finestra di congestione
        df_classification_train['_amplitude_c_cwin'] = df_train['_c_cwin_max'] - df_train['_c_cwin_min']
        # densita dei dati trasmessi rispetto alla dimensione media dei pacchetti
        df_classification_train['_data_density'] = df_train['_c_pkts_data_avg']/df_train['_c_msgsize_count']
    
    
    # standardized dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_classification_train.drop('label', axis=1).to_numpy())
    
    
    # LDA
    lda_col = ['LDA'+str(N) for N in range(len(unique_labels)-1)]

    lda= LDA()
    lda.fit(X_train,df_classification_train['label'])
    
    
    # Transform the training samples
    lda_train_f = lda.transform(X_train)
    lda_train_f = pd.DataFrame(lda_train_f, columns=lda_col)
    lda_train_f['label'] = df_classification_train['label'].to_numpy()
    
    
    ## validation
    # Remove the categorical data
    df_classification_val = df_val.drop(['_c_port', '_s_port', 'c_ip','_tls_session_stat'],axis=1)
    # Remove the unwated features seen in the first task
    df_classification_val = df_classification_val.drop(np.unique(features_to_eliminate),axis=1)
    
    # add the new features
    ADD_Feature = 1
    if ADD_Feature == 1:
        # ampiezza della finestra di congestione
        df_classification_val['_amplitude_c_cwin'] = df_val['_c_cwin_max'] - df_val['_c_cwin_min']
        # densita dei dati trasmessi rispetto alla dimensione media dei pacchetti
        df_classification_val['_data_density'] = df_val['_c_pkts_data_avg']/df_val['_c_msgsize_count']
        
    # standardized dataset
    X_val_f = scaler.transform(df_classification_val.drop('label', axis=1).to_numpy())
    
    # LDA
    # Transform the validation samples
    lda_val_f = lda.transform(X_val_f)
    lda_val_f = pd.DataFrame(lda_val_f, columns=lda_col)
    lda_val_f['label'] = df_classification_val['label'].to_numpy()
    
    
    
    #3-NN classifier WITH LDA
    KNeighborsClassifier_(data_train=lda_train_f.drop(['label'], axis=1).to_numpy(), data_val=lda_val_f.drop(['label'], axis=1).to_numpy(),
                label_train=lda_train_f['label'].to_numpy(), label_val=lda_val_f['label'].to_numpy(),
                title='3-NN classifier WITH LDA ',k=3)
    
    
    # %% TASK 3  - Features Elimination #

    best_hyperParam_DT = []
    best_hyperParam_RF = []
    
    # Decision Tree
    for FLAG_accuracy, ylab in zip([True, False],['Accuracy','F1 score']):
        best_clf_DT = run_DT(X_train_s, y_train,X_val_s,y_val, FLAG_accuracy) 
        DT_feature_importance = best_clf_DT.feature_importances_
        
        # best hyper paramters
        best_hyperParam_DT.append(best_clf_DT.max_depth)
        best_hyperParam_DT.append(best_clf_DT.min_samples_split)
        best_hyperParam_DT.append(best_clf_DT.min_impurity_decrease)
        
        # figura di Future Importance
        features = list(df_train.columns)
        feature_importance = {features[i]: DT_feature_importance[i] for i in range(len(DT_feature_importance))}
        feature_importance_sorted = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1])}
        
        fig, ax = plt.subplots(figsize=(5,4))    
        ax.plot(feature_importance_sorted.values())
        ax.set_title("DT Features Importance")
        ax.set_ylabel("Feature Importance")
        ax.set_xlabel("n° of features")
        plt.tight_layout()
        plt.grid()
        plt.show()
        plt.close()     
        
        
        global_performance_train, global_performance_val = compute_recurrent_feature_elimination(best_clf_DT,df_train,list(feature_importance_sorted.keys()), FLAG_accuracy)
        
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot([i for i in range(len(global_performance_train),0,-1)], global_performance_train, label="Train")
        ax.plot([i for i in range(len(global_performance_train),0,-1)], global_performance_val, label="Validation")
        ax.set_title("Decision Tree")
        ax.set_ylabel(ylab)
        ax.set_xlabel("n° of features")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    
    
    # Random Forest
    for FLAG_accuracy, ylab in zip([True, False],['Accuracy','F1 score']):
        best_clf_RF = run_RF(X_train_s, y_train,X_val_s,y_val, FLAG_accuracy) 
        DT_feature_importance = best_clf_RF.feature_importances_
    
        # best hyper paramters
        best_hyperParam_RF.append(best_clf_RF.max_depth)
        best_hyperParam_RF.append(best_clf_RF.min_samples_split)
        best_hyperParam_RF.append(best_clf_RF.min_impurity_decrease)
        best_hyperParam_RF.append(best_clf_RF.n_estimators)
        
        # figura di Future Importance
        features = list(df_train.columns)
        feature_importance = {features[i]: DT_feature_importance[i] for i in range(len(DT_feature_importance))}
        feature_importance_sorted = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1])}
        
        fig, ax = plt.subplots(figsize=(5,4))    
        ax.plot(feature_importance_sorted.values())
        ax.set_title("RF Features Importance")
        ax.set_ylabel("Feature Importance")
        ax.set_xlabel("n° of features")
        plt.tight_layout()
        plt.grid()
        plt.show()
        plt.close()     
        
        
        global_performance_train, global_performance_val = compute_recurrent_feature_elimination(best_clf_RF,df_train,list(feature_importance_sorted.keys()), FLAG_accuracy)
        
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot([i for i in range(len(global_performance_train),0,-1)], global_performance_train, label="Train")
        ax.plot([i for i in range(len(global_performance_train),0,-1)], global_performance_val, label="Validation")
        ax.set_title("Random Forest")
        ax.set_ylabel(ylab)
        ax.set_xlabel("n° of features")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()
        
        N_useful_features = 12
        
        # bar plot of the most important features used in the clustering task
        n_useful_features = 12
        features_importance_list =  sorted(feature_importance_sorted.items(), key=operator.itemgetter(1), reverse=True)
        useful_features = features_importance_list[:12]
        feature_names = [keys[0] for keys in useful_features]
        feature_importance = [values[1] for values in useful_features]
        
        

        fig, ax = plt.subplots(figsize=(8, 6))                
        ax.bar(range(n_useful_features), feature_importance, color='lightblue', align='center')
        ax.set_xticks(range(n_useful_features), feature_names, rotation=90, fontsize=7)
        ax.set_ylabel('Feature Importance')
        ax.set_title('Top {} Features Importance'.format(n_useful_features))
        plt.tight_layout()
        plt.show()
        
        
    # Data Reduction on the rows
    perc = 5
    percentage = np.linspace(1,99,int(np.floor((100/perc))))
    
    #DT
    for FLAG_accuracy, ylab in zip([True, False],['Accuracy','F1 score']):
        global_performance_train, global_performance_val = recursive_stratified_elimination(best_clf_DT,df_train,percentage,unique_labels, FLAG_accuracy)
        
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(percentage, global_performance_train, label="Train")
        ax.plot(percentage, global_performance_val, label="Validation")
        ax.set_title("Decision Tree")
        ax.set_ylabel(ylab)
        ax.set_xlabel("% of data eliminated")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    #RF
    for FLAG_accuracy, ylab in zip([True, False],['Accuracy','F1 score']):
        global_performance_train, global_performance_val = recursive_stratified_elimination(best_clf_RF,df_train,percentage,unique_labels, FLAG_accuracy)
        
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(percentage, global_performance_train, label="Train")
        ax.plot(percentage, global_performance_val, label="Validation")
        ax.set_title("Random Forest")
        ax.set_ylabel(ylab)
        ax.set_xlabel("% of data eliminated")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    
    # %% Task 4 - Clustering
    
    #taking the most important features
    N=12 #from the previous graph
    
    
    Flag_Task3 = 0
    if Flag_Task3 == 1:
        columns_features=[]
        
        for i in range(0,N):
            c = feature_importance_sorted.popitem()
            columns_features.append(c[0])
    else:
        columns_features = ['_s_msgsize1', '_s_appdataB', '_s_cwin_ini', '_s_cwin_max', '_c_cwin_max', '_s_bytes_uniq', '_s_bytes_all', '_c_mss_max', '_c_pkts_data_std', '_c_pkts_data_avg' , '_s_pkts_data_avg', '_c_msgsize1']
    
    
    df_clustering = df_train[columns_features]
    y = df_train['label']
     
    X_train_cl, X_val_cl, y_train_cl, y_val_cl = train_test_split(df_clustering, y, stratify=y, train_size=.7, random_state=15)

    # standardized dataset
    scaler = StandardScaler()
    scaler.fit(X_train_cl)
    X_train_s_cl, X_val_s_cl = scaler.transform(X_train_cl), scaler.transform(X_val_cl)
    
    ## PCA
    # Get the full PCs set
    pca = PCA(random_state=15, n_components = 10) # per rimuovere gli outliers
    pca.fit(X_train_s_cl)
   
    # Transform Training
    pca_train = pca.transform(X_train_s_cl)
    pca_train = pd.DataFrame(pca_train, columns=[f'PC{i}' for i in range(N)])
    pca_train = pca_train.reset_index(drop=True)
    y_train_cl = y_train_cl.reset_index(drop=True)
    pca_train['label'] = y_train_cl

    # Transform Validation
    pca_val = pca.transform(X_val_s_cl)
    pca_val = pd.DataFrame(pca_val, columns=[f'PC{i}' for i in range(N)])
    pca_val= pca_val.reset_index(drop=True)
    y_val_cl = y_val_cl.reset_index(drop=True)
    pca_val['label'] = y_val_cl
    
    
    
    pca_train_ = pca_train.drop(['label'],axis=1)
    label_train = pca_train['label']
    pca_val_ = pca_val.drop(['label'],axis=1)
    label_val = pca_val['label']
    
     
    
    KM = 0
    if KM == 1:   
            
        Score_KM = pd.DataFrame(columns = ['n_cluster', 'Silhouette', 'Distortion', 'Homogeneity', 'Purity', 'ARI'])
        
        for n_clusters in range(2,34):
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=15)
            predicted_labels = kmeans.fit_predict(pca_train_)
            
            # Distortion
            Distortion = kmeans.inertia_
            
            # Unsupervised metric
            Silhouette  = silhouette_score(pca_train_, predicted_labels)
            
            #Supervised metrics
            Homogeneity = homogeneity_score(label_train, predicted_labels)
            Purity = purity_score(label_train, predicted_labels)
            ARI = adjusted_rand_score(label_train, predicted_labels)
            
            new_row = {'n_cluster':n_clusters,'Silhouette':Silhouette, 'Distortion':Distortion, 'Homogeneity':Homogeneity, 'Purity':Purity, 'ARI':ARI}
    
            Score_KM = Score_KM.append(new_row,ignore_index=True)

        # Get n_clusters leading to the highest silhouette
        Score_KM['Silhouette'] = Score_KM['Silhouette'].astype(float)
        index = Score_KM['Silhouette'].argmax()
        
        best_N_cluster_KM, best_shs = Score_KM.iloc[index][['n_cluster', 'Silhouette']].values
        best_N_cluster_KM = int(best_N_cluster_KM)
        
        # Plot Silhouette score
        plt.figure(figsize=(5, 3.5))
        plt.plot(Score_KM['n_cluster'].to_numpy(), Score_KM['Silhouette'].to_numpy(), marker='o', markersize=5)
        plt.scatter(best_N_cluster_KM, best_shs, color='r', marker='x', s=90)
        plt.grid()
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.xlim(1, 34)
        plt.show()
        
        # Plot Elbow Method
        plt.plot(Score_KM['n_cluster'].to_numpy(), Score_KM['Distortion'].to_numpy(), marker='o')
        plt.xlabel('Number of cluster')
        plt.ylabel('Distortion')
        plt.title('k-Means Elbow Method')
        plt.grid()
        plt.show()
    else:
        best_N_cluster_KM = 25
        
    # recompute for the best number of cluster
    kmeans = KMeans(n_clusters=best_N_cluster_KM,  init='k-means++', n_init=10, max_iter=300, random_state=15)
    y_kmeans = kmeans.fit_predict(pca_train_)
    pca_train['KM cluster'] = y_kmeans
    
    #silhouette per sample
    cluster_labels = np.unique(y_kmeans)
    n_clusters = best_N_cluster_KM
    silhouette_vals = silhouette_samples(pca_train_, y_kmeans, metric='euclidean')
    
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_kmeans == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color= cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), 
                 c_silhouette_vals, height=1.0, edgecolor='none',color=color)
        yticks.append((y_ax_lower + y_ax_upper)/2)
        y_ax_lower += len(c_silhouette_vals)
    
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()
    
    
    # 2D scattering of k-means clusters according to PC0 and PC1
    C = list(mcolors.CSS4_COLORS) 
    M=['o','v','1','s','p','P','*','H','+','x','d','|','D','^','8','_','>','4','3','2','<','X',',','.','1','H','o','v']
 
    N = best_N_cluster_KM # optimal number of cluster
    I = np.arange(0,N,1)
    Col = C[:N]
    M = M[:N]
    L = ['cluster'+' '+ str(n) for n in range(1,N+1)] #controllare il +1
   
    plt.figure(figsize=(20,20))
    for i,c,m,l in zip(I,C,M,L):
        subdf = pca_train[pca_train['KM cluster'] == i]
        plt.scatter(subdf['PC0'],subdf['PC1'],
                s=50, c=c, marker=m, label=l)
        
        plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], 
                s=2000, marker='*', edgecolors='white',linewidth=3, facecolor=c, hatch='|')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid()
    plt.title('k-Means Clusters', fontsize=12)
    plt.xlabel('PC0')
    plt.ylabel('PC1')
    plt.show()
    
    # inertia: Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
    print('Distortion: %.2f' % kmeans.inertia_)
    

    ##  HIERARCHICAL AGGLOMERATIVE CLUSTERING 
   
    AGG = 1
    if AGG == 1:

        # balanced Dataset 
        values=dict()
        
        for label in unique_labels:           
            sub_df_train = pca_train[pca_train['label']==label]
            values[label]=sub_df_train['label'].count() 
            
        minimum = min(values.items(), key=operator.itemgetter(1))[1]
        print(minimum) 
        
   
        # tecnica di campionamento - sottocampionamento delle classi maggioritarie
        num_samples = minimum
        labels = label_train.to_numpy()
        data = pca_train_.to_numpy().astype(np.float32)
        subsampled_data_train = []
        subsampled_labels_train = []
         
        # Effettua un ciclo sulle etichette univoche
        for label in unique_labels:
            # Ottieni gli indici dei campioni corrispondenti all'etichetta corrente
              indices = np.where(labels == label)[0]
             
              # Scegli num_samples campioni casualmente da questi indici
              subsample_indices = np.random.choice(indices, num_samples, replace=False)
             
              # Aggiungi i campioni selezionati alla lista di dati e etichette di output
              subsampled_data_train.extend(data[subsample_indices,:])
              subsampled_labels_train.extend(labels[subsample_indices])
        
        
        
        Score_AGG = pd.DataFrame(columns = ['method','n_cluster','Silhouette', 'Homogeneity', 'Purity', 'ARI'])
        for method in ['single', 'complete', 'average']:
            for N in np.arange(10, 51, 10):
                
                # euclidean is the best one? why?
                agglo = AgglomerativeClustering(n_clusters=N, linkage=method, affinity='euclidean')
        
                # evaluation          
                predicted_labels = agglo.fit_predict(subsampled_data_train)
                 
                # Unsupervised metric
                Silhouette = silhouette_score(subsampled_data_train, predicted_labels)
                # Supervised metrics
                Homogeneity = homogeneity_score(subsampled_labels_train, predicted_labels)
                Purity = purity_score(subsampled_labels_train, predicted_labels)
                ARI = adjusted_rand_score(subsampled_labels_train, predicted_labels)
                
                new_row = {'method':method,'n_cluster':N,'Silhouette':Silhouette, 'Homogeneity':Homogeneity, 'Purity':Purity, 'ARI':ARI}
    
                Score_AGG = Score_AGG.append(new_row, ignore_index=True)
        
        Score_AGG['Silhouette'] = Score_AGG['Silhouette'].astype(float)
        index = Score_AGG['Silhouette'].argmax()
        
        method, n_cluster = Score_AGG.iloc[index][['method', 'n_cluster']].values
        n_cluster = int(n_cluster)
          
        agglo = AgglomerativeClustering(n_clusters=n_cluster, linkage=method, affinity='euclidean')
        predicted_labels = agglo.fit_predict(subsampled_data_train)
        
        Sub_DF_train = pd.DataFrame(subsampled_data_train, columns = pca_train_.columns)
        Sub_DF_train['AGG cluster'] = predicted_labels
        Sub_DF_train['label'] = subsampled_labels_train
    
        # plot  
        C = list(mcolors.CSS4_COLORS) 
        M=['o','v','1','s','p','P','*','H','+','x','d','|','D','^','8','_','>','4','3','2','<','X',',','.','1','H','o','v']
        
        N = n_cluster # optimal number of cluster
        I=np.arange(0,N,1)
        Col=C[N:2*N]
        M = M[:N]
        L = ['cluster'+' '+ str(n) for n in range(1,N+1)]
       
        plt.figure(figsize=(20,20))
        for i,c,m,l in zip(I,C,M,L):
            subdf = Sub_DF_train[Sub_DF_train['AGG cluster'] == i]
            plt.scatter(subdf['PC0'],subdf['PC1'],
                    s=50, c=c, marker=m, label=l)
            
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'Agglomerative Clustering - n_cluster={n_cluster}, linkage={method}, affinity=euclidean')
        plt.grid()
        plt.show()    
   
    
    ## DBSCAN
    DB = 1
    if DB == 1:
        
        Score_DBSCAN = pd.DataFrame(columns = ['metric','eps','mins','Silhouette', 'Homogeneity', 'Purity', 'ARI'])
        
        for metric in ['euclidean','cosine']:        
            for eps in np.arange(.01, .2, .02):
                for mins in range(10,50,10):
                    dbscan = DBSCAN(eps=eps, min_samples=mins, metric=metric)
                    predicted_labels = dbscan.fit_predict(pca_train_)
                    
                    #Unsupervised metric
                    try:
                        Silhouette  = silhouette_score(pca_train_, predicted_labels)                   
                    except:
                        Silhouette = np.nan
                    
                    #Supervised metric
                    try:
                        ARI = adjusted_rand_score(label_train, predicted_labels) 
                    except:
                        ARI = np.nan
                        
                    try:
                        Homogeneity = homogeneity_score(label_train, predicted_labels)
                    except:
                        Homogeneity = np.nan
                    
                    try:
                        Purity = purity_score(label_train, predicted_labels)
                    except:
                        Purity = np.nan
    
                            
                    new_row = {'metric':metric, 'eps':eps, 'mins':mins, 'Silhouette':Silhouette, 'Homogeneity':Homogeneity, 'Purity':Purity, 'ARI':ARI}
    
                    Score_DBSCAN = Score_DBSCAN.append(new_row, ignore_index=True)
                        
    
        # Plot Silhouette
        plt.figure()        
        dbsil = Score_DBSCAN[['eps', 'mins', 'Silhouette','metric']]
        dbsil['eps'] = round(dbsil['eps'],2)
        
        dbsil = dbsil[dbsil['metric'] == 'euclidean']
        dbsil = pd.pivot_table(dbsil, columns='eps', index='mins', values='Silhouette', aggfunc=np.mean)
        sns.heatmap(dbsil, cmap='Blues', cbar_kws={'label':'Silhouette score'}, 
                    linewidths=.005)
        plt.gca().invert_yaxis()
        plt.title('Silhouette, metric = euclidean')
        plt.show()
        
        plt.figure()
        dbsil = Score_DBSCAN[['eps', 'mins', 'Silhouette','metric']]
        dbsil['eps'] = round(dbsil['eps'],2)
        
        dbsil = dbsil[dbsil['metric'] == 'cosine']
        dbsil['eps'] = round(dbsil['eps'],2)
        dbsil = pd.pivot_table(dbsil, columns='eps', index='mins', values='Silhouette', aggfunc=np.mean)
        sns.heatmap(dbsil, cmap='Blues', cbar_kws={'label':'Silhouette score'}, 
                    linewidths=.005)
        plt.gca().invert_yaxis()
        plt.title('Silhouette, metric = cosine')
        plt.show()
            
        # Plot Ari
        plt.figure()
        dbari = Score_DBSCAN[['eps', 'mins', 'ARI','metric']]
        dbari['eps'] = round(dbari['eps'],2)
        
        dbari = dbari[dbari['metric'] == 'euclidean']
        dbari = pd.pivot_table(dbari, columns='eps', index='mins', values='ARI', aggfunc=np.mean) 
        sns.heatmap(dbari, cmap='Purples', cbar_kws={'label':'ARI score'},linewidths=.005)
        plt.gca().invert_yaxis()
        plt.title('Ari, metric = euclidean')
        plt.show()
        
        plt.figure()
        dbari = Score_DBSCAN[['eps', 'mins', 'ARI','metric']]
        dbari['eps'] = round(dbari['eps'],2)
        
        dbari = dbari[dbari['metric'] == 'cosine']
        dbari = pd.pivot_table(dbari, columns='eps', index='mins', values='ARI', aggfunc=np.mean) 
        sns.heatmap(dbari, cmap='Purples', cbar_kws={'label':'ARI score'},linewidths=.005)
        plt.gca().invert_yaxis()
        plt.title('Ari, metric = cosine')
        plt.show()
            
        
        # best hyper-parameter
        Score_DBSCAN['Silhouette'] = Score_DBSCAN['Silhouette'].astype(float)
        index = Score_DBSCAN['Silhouette'].argmax()  
        metric, eps, M = Score_DBSCAN.iloc[index][['metric','eps','mins']].values
        M = int(M)
    
        dbscan = DBSCAN(eps=eps, min_samples=M, metric=metric)
        predicted_labels = dbscan.fit_predict(pca_train_)
        pca_train['DBSCAN cluster'] = predicted_labels
        
    
        # Plot Confusion Matrix
        plt.figure(figsize=(20, 8))
        confusion_val = confusion_matrix(np.asarray(label_train), np.asarray(predicted_labels))
        sns.heatmap(confusion_val, cmap='Blues', annot=True, vmin=0, vmax=100, cbar_kws={'label':'Occurrences'})
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.grid()
    
    
## Clusters Evaluation on Validation Set
    
    Validation_Score = pd.DataFrame(columns = ['algorithm', 'Silhouette', 'Homogeneity', 'Purity', 'ARI'])
    C = colors = [ "#2980b9", "#3498db", "#5DADE2", "#85C1E9", "#AED6F1", "#D6EAF8", "#e74c3c", "#e67e22", "#f39c12", "#f1c40f", "#f9e79f", 
                  "#fcf3cf", "#2ecc71", "#27ae60", "#58D68D", "#82E0AA", "#ABEBC6", "#D4EFDF", "#8e44ad", "#9b59b6", "#BB8FCE", "#D2B4DE", 
                  "#E8DAEF", "#F4ECF7", "#8E44AD", "#F39C12" ]
    
    ##K-Means
    predicted_labels = kmeans.predict(pca_val_)
    pca_val['KM cluster'] = predicted_labels
    
    # Unsupervised metric
    Silhouette  = silhouette_score(pca_val_, predicted_labels)
    #Supervised metrics
    Homogeneity = homogeneity_score(label_val, predicted_labels)
    Purity = purity_score(label_val, predicted_labels)
    ARI = adjusted_rand_score(label_val, predicted_labels)
    
    new_row = {'algorithm':'K_Means', 'Silhouette':Silhouette, 'Homogeneity':Homogeneity,'Purity':Purity, 'ARI':ARI}
    Validation_Score = Validation_Score.append(new_row, ignore_index=True)
    
    
  
    I=np.arange(0,best_N_cluster_KM,1)
    markers=['o','v','1','s','p','P','*','H','+','x','d','|','D','^','8','_','>','4','3','2','<','X',',','.','1','H']
    L = ['cluster'+' '+ str(n) for n in range(1,N+1)]
    
    fig, axis = plt.subplots(2,2, figsize=(22,22))

    for i,c,m,l in zip(I,C[:N],markers,L[:best_N_cluster_KM]):
         subdf =pca_val[pca_val['KM cluster'] == i]
         axis[0,0].scatter(subdf['PC0'],subdf['PC1'],
                 s=50, c=c, marker=m, label=l)
         
         axis[0,0].scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], 
                 s=1000, marker='*', edgecolors='black',linewidth=1, facecolor=c, hatch='|||')
         
    axis[0,0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
    axis[0,0].grid()
    axis[0,0].set_title('k-Means', fontsize=18)

    ##GT

    for i,c,m,l in zip(unique_labels,C[:len(np.unique(unique_labels))],markers,unique_labels):
         subdf =pca_val[pca_val['label'] == i]
         axis[0,1].scatter(subdf['PC0'],subdf['PC1'],
                 s=50, c=c, marker=m, label=l)
        
    axis[0,1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
    axis[0,1].grid()
    axis[0,1].set_title('GT', fontsize=18)
    
    ##AG
    
    # balanced Dataset 
    values=dict()
    
    for label in unique_labels:           
        sub_df_val = pca_val[pca_val['label']==label]
        values[label]=sub_df_val['label'].count() 
        
    minimum = min(values.items(), key=operator.itemgetter(1))[1]
    print(minimum) 
    
    # tecnica di campionamento - sottocampionamento delle classi maggioritarie
    num_samples = minimum
    labels = label_val.to_numpy()
    data = pca_val_.to_numpy().astype(np.float32)
    subsampled_data_val = []
    subsampled_labels_val = []
     
    # Effettua un ciclo sulle etichette univoche
    for label in unique_labels:
        # Ottieni gli indici dei campioni corrispondenti all'etichetta corrente
          indices = np.where(labels == label)[0]
         
          # Scegli num_samples campioni casualmente da questi indici
          subsample_indices = np.random.choice(indices, num_samples, replace=False)
         
          # Aggiungi i campioni selezionati alla lista di dati e etichette di output
          subsampled_data_val.extend(data[subsample_indices,:])
          subsampled_labels_val.extend(labels[subsample_indices])
          
    predicted_labels = agglo.fit_predict(subsampled_data_val)
    
    Sub_DF_val = pd.DataFrame(subsampled_data_val, columns = pca_val_.columns)
    Sub_DF_val['AGG cluster'] = predicted_labels
    Sub_DF_val['label'] = subsampled_labels_val
    
    # Unsupervised metric
    Silhouette = silhouette_score(subsampled_data_val, predicted_labels)
    # Supervised metrics
    Homogeneity = homogeneity_score(subsampled_labels_val, predicted_labels)
    Purity = purity_score(subsampled_labels_val, predicted_labels)
    ARI = adjusted_rand_score(subsampled_labels_val, predicted_labels)
    
    new_row = {'algorithm':'Aglomerative', 'Silhouette':Silhouette, 'Homogeneity':Homogeneity,'Purity':Purity, 'ARI':ARI}
    Validation_Score = Validation_Score.append(new_row, ignore_index=True)
    
    
    for i,c,m,l in zip(I,C[:len(np.unique(predicted_labels))],markers,L[:agglo.n_clusters_]):
          subdf =Sub_DF_val[Sub_DF_val['AGG cluster'] == i]
          axis[1,0].scatter(subdf['PC0'],subdf['PC1'],
                  s=50, c=c, marker=m, label=l)
          
    axis[1,0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
    axis[1,0].grid()
    axis[1,0].set_title('Agglomerative Clustering - with Balanced Dataset', fontsize=18)
    
    ##DBSCAN
    predicted_labels = dbscan.fit_predict(pca_val_)
    pca_val['DBSCAN cluster'] = predicted_labels
    
    # Unsupervised metric
    Silhouette  = silhouette_score(pca_val_, predicted_labels)
    #Supervised metrics
    Homogeneity = homogeneity_score(label_val, predicted_labels)
    Purity = purity_score(label_val, predicted_labels)
    ARI = adjusted_rand_score(label_val, predicted_labels)
    
    new_row = {'algorithm':'DBSCAN', 'Silhouette':Silhouette, 'Homogeneity':Homogeneity,'Purity':Purity, 'ARI':ARI}
    Validation_Score = Validation_Score.append(new_row, ignore_index=True)

    for i,c,m,l in zip(predicted_labels,C[:len(np.unique(predicted_labels))],markers,L[:len(np.unique(predicted_labels))]):
          subdf =pca_val[pca_val['DBSCAN cluster'] == i]
          plt.scatter(subdf['PC0'],subdf['PC1'],
                  s=50, c=c, marker=m, label=l)
          
    axis[1,1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
    axis[1,1].grid()
    axis[1,1].set_title('DBSCAN', fontsize=18)
    plt.show()  
   
    ## Final evaluation
    df_clustering_val = df_val.drop(['_c_port', '_s_port','c_ip'],axis=1)
    
    # #taking the most important features
    N=12 #from the previous graph
    
    
    Flag_Task3 = 0
    if Flag_Task3 == 1:
        columns_features=[]
        
        for i in range(0,N):
            c = feature_importance_sorted.popitem()
            columns_features.append(c[0])
    else:
        columns_features = ['_s_msgsize1', '_s_appdataB', '_s_cwin_ini', '_s_cwin_max', '_c_cwin_max', '_s_bytes_uniq', '_s_bytes_all', '_c_mss_max', '_c_pkts_data_std', '_c_pkts_data_avg' , '_s_pkts_data_avg', '_c_msgsize1']
    
    # training
    df_clustering_train = df_train[columns_features]
    df_clustering_train['label'] = df_train['label']
    
    X = df_clustering_train[columns_features].to_numpy()
    y_train = df_clustering_train['label'].to_numpy().flatten()
     
    # standardized dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    
    ## PCA
    # Get the full PCs set
    pca = PCA(random_state=15, n_components = 10) # per rimuovere gli outliers
    X_train_pca = pca.fit_transform(X_train)
    
    # validation
    df_clustering_val_2 = df_clustering_val[columns_features]
    df_clustering_val_2['label'] = df_val['label']
    
    X = df_clustering_val_2[columns_features].to_numpy()
    y_val = df_clustering_val_2['label'].to_numpy().flatten()
     
    # standardized dataset
    X_val = scaler.transform(X)
    
    ## PCA
    X_val_pca = pca.transform(X_val)
    
    
    # K-Means
    kmeans = KMeans(n_clusters=best_N_cluster_KM,  init='k-means++', n_init=10, max_iter=300, random_state=15)
    kmeans.fit(X_train_pca)
    
    
    predicted_labels = kmeans.predict(X_val_pca)
    
    KM_dt = pd.DataFrame(X_val_pca, columns= ['PC'+str(N) for N in range(X_val_pca.shape[1])])
    KM_dt['KM cluster'] = predicted_labels
    KM_dt['True Label'] = y_val
    
    # Unsupervised metric
    Silhouette  = silhouette_score(X_val_pca, predicted_labels)
    #Supervised metrics
    Homogeneity = homogeneity_score(y, predicted_labels)
    Purity = purity_score(y, predicted_labels)
    ARI = adjusted_rand_score(y, predicted_labels)
    
    new_row = {'algorithm':'Final K_Means', 'Silhouette':Silhouette, 'Homogeneity':Homogeneity,'Purity':Purity, 'ARI':ARI}
    Validation_Score = Validation_Score.append(new_row, ignore_index=True)
    
    
    I=np.arange(0,best_N_cluster_KM,1)
    markers=['o','v','1','s','p','P','*','H','+','x','d','|','D','^','8','_','>','4','3','2','<','X',',','.','1','H']
    L = ['cluster'+' '+ str(n) for n in range(1,best_N_cluster_KM+1)]
    
    # Plot
    fig, axis = plt.subplots(1,2, figsize=(30,10))
 
    for i,c,m,l in zip(I,C[:best_N_cluster_KM],markers,L[:best_N_cluster_KM]):
         subdf =KM_dt[KM_dt['KM cluster'] == i]
         axis[0].scatter(subdf['PC0'],subdf['PC1'],
                 s=50, c=c, marker=m, label=l)
         
         axis[0].scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], 
                 s=1000, marker='*', edgecolors='black',linewidth=1, facecolor=c, hatch='|||')
         
    axis[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
    axis[0].grid()
    axis[0].set_title('k-Means', fontsize=18)
    axis[0].set_xlabel('PC0')
    axis[0].set_ylabel('PC1')
    
    
    #GT  
    for c,m,l in zip(C[:len(unique_labels)],markers,unique_labels):
         subdf =KM_dt[KM_dt['True Label'] == l]
         axis[1].scatter(subdf['PC0'],subdf['PC1'],
                 s=50, c=c, marker=m, label=l)
     
    axis[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
    axis[1].grid()
    axis[1].set_title('GT', fontsize=18)
    axis[1].set_xlabel('PC0')
    axis[1].set_ylabel('PC1')
    
    plt.suptitle('Validation Result', fontsize=20)
    
