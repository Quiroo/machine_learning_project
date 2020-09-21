# 7)*Pending* Translate columns to English 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Graphs():
    def BoxplotGender (X):
        g = sns.catplot(x="victima_genero",y="victima_edad",data=X, kind="boxen", height = 5 ,aspect=3,palette = "muted")
        g.despine(left=True)
        g.set_xticklabels(rotation=0)
        plt.xlabel("Victim Gender")
        plt.ylabel("Age")
        plt.title('Boxplot Victim Geneder VS Age',size = 12)
        return plt.show()

    def AgePlot (X):
        lavf_edades2 = X
        convert_dict = {'victima_cantidad': int, 'victima_edad': int} 
        lavf_edades2 = lavf_edades2.astype(convert_dict)
        victima_edades2 = lavf_edades2.iloc[:,4]
        victima_edades2.describe()
    
        plt.figure(figsize=(20,5))
        a = sns.countplot(x="victima_edad",data=lavf_edades2)
        a.set_xticklabels(a.get_xticklabels(), rotation=90)
        plt.xlabel("Age of victims")
        plt.ylabel("Amount")
        plt.title('Age distribution',size = 20)

        plt.figure(figsize=(20,5))
        sns.kdeplot(victima_edades2, color="Blue", shade = True)
        plt.xlabel("Age of victims")
        plt.ylabel("Frecuency")


        plt.figure(figsize=(20,5))
        sns.countplot(x="victima_rango_etario",data=lavf_edades2)
        plt.xlabel("Range age of victims")
        plt.ylabel("Amount")
        return plt.show()

    def AgePlot2 (X):
        lavf_edades2 = X
        convert_dict = {'victima_cantidad': int, 'victima_edad': int} 
        lavf_edades2 = lavf_edades2.astype(convert_dict)
        victima_edades2 = lavf_edades2.iloc[:,4]
        victima_edades2.describe()
    
        plt.figure(figsize=(20,5))
        a = sns.countplot(x="victima_edad",data=lavf_edades2)
        a.set_xticklabels(a.get_xticklabels(), rotation=90)
        plt.xlabel("Age of victims")
        plt.ylabel("Amount")
        plt.title('Age distribution',size = 20)

        plt.figure(figsize=(20,5))
        sns.kdeplot(victima_edades2, color="Blue", shade = True)
        plt.xlabel("Age of victims")
        plt.ylabel("Frecuency")

        return plt.show()

    
    def TimePlot(X):
    
        ay = X['hora'].value_counts()
        ax = X['hora'].unique()
        ay2 = X['hora_sola'].value_counts()
        ax2 = X['hora_sola'].unique()
        horasola2 = {'Cantidad de llamados': ay2}
        horasoladt2 = pd.DataFrame(data=horasola2)
        horasoladt2 = horasoladt2.reset_index()
        horacompleta2 = {'Cantidad de llamados': ay}
        horacompletadt2 = pd.DataFrame(data=horacompleta2)
        horacompletadt2 = horacompletadt2.reset_index()
    
        plt.figure(figsize=(15,5))
        asns = sns.lineplot(x="index", y="Cantidad de llamados", data=horacompletadt2)
        asns.set(xlabel="Hour", ylabel="Call Amount")
        plt.title('Call Distribution during day',size = 20)

        plt.figure(figsize=(15,5))
        asns = sns.lineplot(x="index",y="Cantidad de llamados", data=horasoladt2)
        asns.set(xlabel="Hour", ylabel="Call Amount")
        asns.set(xticks=ax2)
    
        plt.figure(figsize=(15,5))
        g = sns.kdeplot(X['hora_sola'], color="Green", shade = True)
        plt.xlabel("Hour")
        plt.ylabel("Frecuency")
        plt.title('Calls Distribucion in 24hs',size = 20)
        g.set(xticks=ax2)
        return plt.show()

    def WeekdayPlot(X):
        df_momentos = X.groupby(['día', 'momento_del_dia']).size().reset_index().pivot(columns='día', index='momento_del_dia', values=0)
        df_momentos_inversa = X.groupby(['día', 'momento_del_dia']).size().reset_index().pivot(columns='momento_del_dia', index='día', values=0)
        df_momentos_inversa.plot(kind='bar', stacked=True, figsize=(12,5), colormap="viridis")
        plt.xlabel("By week day")
        plt.ylabel("Amount")
        return plt.show()

    def DaytimePlot(X):
        plt.figure(figsize=(10,5))
        plt.title('Parts of the day',size = 20)
        sns.countplot(x="momento_del_dia",data=X)
        plt.xlabel("Parts of the day")
        plt.ylabel("Amount")

#    plt.figure(figsize=(10,5))
#    plt.title('Cantidad de llamados por día de la semana',size = 20)
#    sns.countplot(x="día",data=X)
#    plt.xlabel("Día de la semana")
#    plt.ylabel("Cantidad")
        return plt.show()

    def PearsonCorrelation(X):
        plt.figure(figsize=(25,15))
        sns.set_style("white")
        sns.set_context("talk")
        sns.set_style("ticks")
        sns.heatmap(X.corr(), annot=True, fmt='.2g')
        plt.title("Pearson Linear Correlation between features")
        return plt.show()

    def LRdistplotSFS(ytest,ypred_lr_sfs):
        sns.distplot(ytest, color="b", label='Test')
        sns.distplot(ypred_lr_sfs, color="g").set(title='Linear Regression without Feature Selection', xlabel='Victims age')
        return plt.show()
    
    def LRdistplotTr(ytest,ypred_lr_var):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_lr_var, color="g").set(title='Linear Regression with Threshole',xlabel='Victims age')
        plt.show()
        
    def LRdistplotLs(ytest,ypred_lr_lasso):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_lr_lasso, color="g").set(title='Linear Regression with Lasso',xlabel='Victims age')
        plt.show()
    
    def KNNdistplotSFS(ytest,ypred_knn_sfs):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_knn_sfs, color="g")
        plt.show()
        
    def KNNdistplotTh(ytest,ypred_knn_var):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_knn_var, color="g")
        plt.show()
        
    def KNNdistplotLs(ytest, ypred_knn_lasso):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_knn_lasso, color="g")
        plt.show()
        
    def SRVdistplotSFS(ytest,ypred_svr_l_sfs):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_svr_l_sfs, color="g")
        plt.show()
        
    def SRVdistplotTh(ytest, ypred_svr_l_var):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_svr_l_var, color="g")
        plt.show()
        
    def SRVdistplotLs(ytest, ypred_svr_l_lasso):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_svr_l_lasso, color="g")
        plt.show()
        
    def GSdistplotSFS(ytest, ypred_svr_g_sfs):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_svr_g_sfs, color="g")
        plt.show()
        
    def GSdistplotTh(ytest, ypred_svr_g_var):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_svr_g_var, color="g")
        plt.show()
        
    def GSdistplotLs(ytest, ypred_svr_g_lasso):
        sns.distplot(ytest, color="b")
        sns.distplot(ypred_svr_g_lasso, color="g")
        plt.show()
        
    def ROC_KNN_Lasso(fpr1, tpr1, auc_knn_l_cl):
        plt.plot(fpr1, tpr1, lw=2, alpha=0.7 , label = 'ROC curve', color = 'b')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',label='Luck', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(False)
        plt.legend(loc="lower right")
        plt.title('ROC curve with KNN classifier + Lasso = '+str(np.round(auc_knn_l_cl,4)))
        plt.show()    
        
    def ConfussionMatrix_KNN_Lasso(ytest, ypred_knn_test_l_cl):
        cm_knn_l_cl = confusion_matrix(ytest, ypred_knn_test_l_cl)
        df_knn_l_cl = pd.DataFrame(cm_knn_l_cl, index = ['Intervention', 'No Intervention'], columns = ['Intervention predicted', 'No Intervention predicted'])
        plt.figure(figsize = (6,4))
        sns.heatmap(df_knn_l_cl, annot=True, fmt='g')
        plt.title('Classification Confusion matrix with Lasso')
        plt.show()
        
    def ROC_SVM_Lasso(fpr2, tpr2, auc_svm_l):
        plt.plot(fpr2, tpr2, lw=2, alpha=0.7 , label = 'ROC curve', color = 'b')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',label='Luck', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(False)
        plt.legend(loc="lower right")
        plt.title('ROC curve with LR classifier + Lasso = '+str(np.round(auc_svm_l,4)))
        plt.show()
    
    
    def ConfussionMatrix_SVM_Lasso(ytest2, ypred_svm_test):    
        cm_svm = confusion_matrix(ytest2, ypred_svm_test)
        df_svm = pd.DataFrame(cm_svm, index = ['Intervention', 'No Intervencion'], columns = ['Intervention predicted', 'No Intervencion predicted'])
        plt.figure(figsize = (6,4))
        sns.heatmap(df_svm, annot=True, fmt='g')
        plt.title('Classification Confusion matrix con Lasso')
        plt.show()
        
        
        
    
class Text():
    def FeatureSelection(T,L):
            print("The Features selected by THRESHOLD method [8]")
            print(T.columns)
            print()
            print("The Features selected by LASSO method [12]")
            print(L.columns)
            print()

    def Comparison(LR_Stats, KNN_Stats, SVR_Stats, GS_Stats):
            LR_NoFS = LR_Stats[0]
            LR_Thershole = LR_Stats[1]                  
            LR_Lasso = LR_Stats[2]            
            KNN_NoFS = KNN_Stats[0]            
            KNN_Threshole = KNN_Stats[1]            
            KNN_Lasso = KNN_Stats[2]            
            SVR_NoFS = SVR_Stats[0]            
            SVR_Threshole = SVR_Stats[1]            
            SVR_Lasso = SVR_Stats[2]            
            GS_NoFS = GS_Stats[0]            
            GS_Threshole = GS_Stats[1]            
            GS_Lasso = GS_Stats[2]
            
            comparison_data = {'LR-NFS': LR_NoFS, 'LR-THR': LR_Thershole, 'LR-LAS': LR_Lasso, 'KNN-NFS': KNN_NoFS, 'KNN-THR': KNN_Threshole, 'KNN-LAS': KNN_Lasso, 'SVR-NFS': SVR_NoFS, 'SVR-THR': SVR_Threshole, 'SVR-LAS': SVR_Lasso, 'SVR-G-NFS': GS_NoFS, 'SVR-G-THR': GS_Threshole, 'SVR-G-LAS': GS_Lasso}
          
            summary = pd.DataFrame.from_dict(data = comparison_data, orient='index', columns=['RMSE', 'MSE', 'MAE'])
            return summary            
 
    def Chosen(): 
            print("The models selected to advance with the prediction of age are:")
            print("- KNN with Lasso")
            print("- Gaussian SVR without Feature Selection")
            print("These models are chosen, despite not having the best performance according to their errors. Graphically their curve and histogram are the most faithful to the test one")
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
