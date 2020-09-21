import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
lr_model = LinearRegression()
svreg = SVR()


def LabelEncoding(X):
        le = preprocessing.LabelEncoder()
        lavf_preselection = X.drop(['llamado_fecha_hora','hora'], axis=1)
        lavf_preselection['llamante_descripcion'] = le.fit_transform(lavf_preselection['llamante_descripcion'])
        lavf_preselection['llamante_genero'] = le.fit_transform(lavf_preselection['llamante_genero'])
        lavf_preselection['llamante_vinculo_ninios_presentes'] = le.fit_transform(lavf_preselection['llamante_vinculo_ninios_presentes'])
        lavf_preselection['violencia_tipo'] = le.fit_transform(lavf_preselection['violencia_tipo'])
        lavf_preselection['victima_rango_etario'] = le.fit_transform(lavf_preselection['victima_rango_etario'])
        lavf_preselection['victima_genero'] = le.fit_transform(lavf_preselection['victima_genero'])
        lavf_preselection['agresor_genero'] = le.fit_transform(lavf_preselection['agresor_genero'])
        lavf_preselection['agresor_relacion_victima'] = le.fit_transform(lavf_preselection['agresor_relacion_victima'])
        lavf_preselection['momento_del_dia'] = le.fit_transform(lavf_preselection['momento_del_dia'])
        lavf_preselection['llamado_derivacion'] = le.fit_transform(lavf_preselection['llamado_derivacion'])
        lavf_preselection['día'] = le.fit_transform(lavf_preselection['día'])
        print("*CORREGIR TEXTO:* Note the remarkable inversely linear relationship between the gender of the victim and the gender of the aggressor of [-0.39]")
        return lavf_preselection

def LabelEncoding2(X):
        le = preprocessing.LabelEncoder()
        lavf_preselectionC = X.drop(['llamado_fecha_hora','hora'], axis=1)
        lavf_preselectionC['llamante_descripcion'] = le.fit_transform(lavf_preselectionC['llamante_descripcion'])
        lavf_preselectionC['llamante_genero'] = le.fit_transform(lavf_preselectionC['llamante_genero'])
        lavf_preselectionC['llamante_vinculo_ninios_presentes'] = le.fit_transform(lavf_preselectionC['llamante_vinculo_ninios_presentes'])
        lavf_preselectionC['violencia_tipo'] = le.fit_transform(lavf_preselectionC['violencia_tipo'])
        lavf_preselectionC['victima_rango_etario'] = le.fit_transform(lavf_preselectionC['victima_rango_etario'])
        lavf_preselectionC['victima_genero'] = le.fit_transform(lavf_preselectionC['victima_genero'])
        lavf_preselectionC['agresor_genero'] = le.fit_transform(lavf_preselectionC['agresor_genero'])
        lavf_preselectionC['agresor_relacion_victima'] = le.fit_transform(lavf_preselectionC['agresor_relacion_victima'])
        lavf_preselectionC['momento_del_dia'] = le.fit_transform(lavf_preselectionC['momento_del_dia'])
        lavf_preselectionC['llamado_derivacion'] = le.fit_transform(lavf_preselectionC['llamado_derivacion'])
        lavf_preselectionC['día'] = le.fit_transform(lavf_preselectionC['día'])            
        return lavf_preselectionC
    
def AgeEncoding(age_no_data,xtrain):
        le = preprocessing.LabelEncoder()
        x_sindatos = age_no_data.drop(['victima_edad','llamado_fecha_hora','hora'], axis=1)
        x_sindatos['llamante_descripcion'] = le.fit_transform(x_sindatos['llamante_descripcion'])
        x_sindatos['llamante_genero'] = le.fit_transform(x_sindatos['llamante_genero'])
        x_sindatos['llamante_vinculo_ninios_presentes'] = le.fit_transform(x_sindatos['llamante_vinculo_ninios_presentes'])
        x_sindatos['violencia_tipo'] = le.fit_transform(x_sindatos['violencia_tipo'])
        x_sindatos['victima_rango_etario'] = le.fit_transform(x_sindatos['victima_rango_etario'])
        x_sindatos['victima_genero'] = le.fit_transform(x_sindatos['victima_genero'])
        x_sindatos['agresor_genero'] = le.fit_transform(x_sindatos['agresor_genero'])
        x_sindatos['agresor_relacion_victima'] = le.fit_transform(x_sindatos['agresor_relacion_victima'])
        x_sindatos['momento_del_dia'] = le.fit_transform(x_sindatos['momento_del_dia'])
        x_sindatos['llamado_derivacion'] = le.fit_transform(x_sindatos['llamado_derivacion'])
        x_sindatos['día'] = le.fit_transform(x_sindatos['día'])
        scaler = preprocessing.StandardScaler().fit(xtrain)
        x_nodata_scal = scaler.transform(x_sindatos)
        return x_nodata_scal
   

#def Split(X):
#    x = X.drop(['victima_edad'], axis=1)
#    y = X.iloc[:,4]
#    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4)
#    scaler = preprocessing.StandardScaler().fit(xtrain)
#    xtrain_scal = scaler.transform(xtrain)
#    xtest_scal = scaler.transform(xtest)
    

#def Threshole():
#    thresh = 0.5
#    xtrain_scal_var = xtrain.iloc[:,(np.std(xtrain)>np.quantile(np.std(xtrain), thresh)).values]
#    xtest_scal_var = xtest.iloc[:,(np.std(xtrain)>np.quantile(np.std(xtrain), thresh)).values]
#print("Cantidad de Feautures " + str(np.shape(xtrain_scal_var)[1]))



#Logistic regression model
def LR(xtrain_scal, 
       xtest_scal, 
       xtrain_scal_var,
       xtest_scal_var,
       xtrain_scal_lasso,
       xtest_scal_lasso,
       ytrain,
       ytest,
       lasso_cols):

#Without Feature Selection
        lr_model.fit(xtrain_scal,ytrain)
        ypred_lr_sfs = lr_model.predict(xtest_scal)
        RMSE_lr_sfs = np.sqrt(mean_squared_error(ytest, ypred_lr_sfs))
        MSE_lr_sfs = mean_squared_error(ytest, ypred_lr_sfs)
        MAE_lr_sfs = mean_absolute_error(ytest, ypred_lr_sfs)
        LR_NoFS = [RMSE_lr_sfs, MSE_lr_sfs, MAE_lr_sfs] 
        
#With Threshole
        lr_model.fit(xtrain_scal_var,ytrain)
        ypred_lr_var = lr_model.predict(xtest_scal_var)
        RMSE_lr_var = np.sqrt(mean_squared_error(ytest, ypred_lr_var))
        MSE_lr_var = mean_squared_error(ytest, ypred_lr_var)
        MAE_lr_var = mean_absolute_error(ytest, ypred_lr_var)
        LR_Thershole = [RMSE_lr_var, MSE_lr_var, MAE_lr_var]

#With Lasso
        lr_model.fit(xtrain_scal_lasso,ytrain)
        ypred_lr_lasso = lr_model.predict(xtest_scal_lasso)
        RMSE_lr_lasso = np.sqrt(mean_squared_error(ytest, ypred_lr_lasso))
        MSE_lr_lasso = mean_squared_error(ytest, ypred_lr_lasso)
        MAE_lr_lasso = mean_absolute_error(ytest, ypred_lr_lasso)
        LR_Lasso = [RMSE_lr_lasso, MSE_lr_lasso, MAE_lr_lasso]    
        
        LR_Stats = [LR_NoFS, LR_Thershole, LR_Lasso]
        return ypred_lr_sfs, ypred_lr_var,ypred_lr_lasso, LR_Stats 
 
 #KNN regression model    
def KNN(xtrain_scal,
        xtest_scal,
        xtrain_scal_var,
        xtest_scal_var,
        xtrain_scal_lasso,
        xtest_scal_lasso,
        ytrain,
        ytest):
#KNN Parameters    
        neigh = KNeighborsRegressor(weights = "distance")
        parameters_k = np.arange(2,41,2)
        parameters_knn = [{'n_neighbors': parameters_k}]
        knn_model = GridSearchCV(neigh, parameters_knn, refit = True, cv=5, verbose=True)
#Without Feature Selection    
        knn_model.fit(xtrain_scal, ytrain)
        ypred_knn_sfs = knn_model.predict(xtest_scal)
        RMSE_knn_sfs = np.sqrt(mean_squared_error(ytest, ypred_knn_sfs))
        MSE_knn_sfs = mean_squared_error(ytest, ypred_knn_sfs)
        MAE_knn_sfs = mean_absolute_error(ytest, ypred_knn_sfs)
        KNN_NoFS = [RMSE_knn_sfs, MSE_knn_sfs, MAE_knn_sfs]
#With Threshole
        knn_model.fit(xtrain_scal_var, ytrain)
        ypred_knn_var = knn_model.predict(xtest_scal_var)
        RMSE_knn_var = np.sqrt(mean_squared_error(ytest, ypred_knn_var))
        MSE_knn_var = mean_squared_error(ytest, ypred_knn_var)
        MAE_knn_var = mean_absolute_error(ytest, ypred_knn_var)
        KNN_Threshole = [RMSE_knn_var, MSE_knn_var, MAE_knn_var] 
#With Lasso       
        knn_model.fit(xtrain_scal_lasso, ytrain)
        ypred_knn_lasso = knn_model.predict(xtest_scal_lasso)
        RMSE_knn_lasso = np.sqrt(mean_squared_error(ytest, ypred_knn_lasso))
        MSE_knn_lasso = mean_squared_error(ytest, ypred_knn_lasso)
        MAE_knn_lasso = mean_absolute_error(ytest, ypred_knn_lasso)
        KNN_Lasso = [RMSE_knn_lasso, MSE_knn_lasso, MAE_knn_lasso]
        
        KNN_Stats = [KNN_NoFS, KNN_Threshole, KNN_Lasso]
        return ypred_knn_sfs, ypred_knn_var, ypred_knn_lasso, KNN_Stats
        
#SRV model        
def SVR(xtrain_scal,
        xtest_scal,
        xtrain_scal_var,
        xtest_scal_var,
        xtrain_scal_lasso,
        xtest_scal_lasso,
        ytrain,
        ytest):
#Without Feature Selection  
        svr_model_lin.fit(xtrain_scal,ytrain)
        ypred_svr_l_sfs = svr_model_lin.predict(xtest_scal)
        RMSE_svr_l_sfs = np.sqrt(mean_squared_error(ytest, ypred_svr_l_sfs))
        MSE_svr_l_sfs = mean_squared_error(ytest, ypred_svr_l_sfs)
        MAE_svr_l_sfs = mean_absolute_error(ytest, ypred_svr_l_sfs)
        SVR_NoFS = [RMSE_svr_l_sfs, MSE_svr_l_sfs, MAE_svr_l_sfs]
#With Threshole    
        svr_model_lin.fit(xtrain_scal_var,ytrain)
        ypred_svr_l_var = svr_model_lin.predict(xtest_scal_var)
        RMSE_svr_l_var = np.sqrt(mean_squared_error(ytest, ypred_svr_l_var))
        MSE_svr_l_var = mean_squared_error(ytest, ypred_svr_l_var)
        MAE_svr_l_var = mean_absolute_error(ytest, ypred_svr_l_var) 
        SVR_Threshole = [RMSE_svr_l_var, MSE_svr_l_var, MAE_svr_l_var]
#With Lasso          
        svr_model_lin.fit(xtrain_scal_lasso,ytrain)
        ypred_svr_l_lasso = svr_model_lin.predict(xtest_scal_lasso)
        RMSE_svr_l_lasso = np.sqrt(mean_squared_error(ytest, ypred_svr_l_lasso))    
        MSE_svr_l_lasso = mean_squared_error(ytest, ypred_svr_l_lasso)    
        MAE_svr_l_lasso = mean_absolute_error(ytest, ypred_svr_l_lasso)    
        SVR_Lasso = [RMSE_svr_l_lasso, MSE_svr_l_lasso, MAE_svr_l_lasso]
    
        SVR_Stats = [SVR_NoFS, SVR_Threshole, SVR_Lasso]
        return ypred_svr_l_sfs, ypred_svr_l_var, ypred_svr_l_lasso, SVR_Stats    
        
#Gaussiano
def Gauss(xtrain_scal,
        xtest_scal,
        xtrain_scal_var,
        xtest_scal_var,
        xtrain_scal_lasso,
        xtest_scal_lasso,
        ytrain,
        ytest):
#Gauss Parameters
#    parameters_svr_lin = [{'kernel':['linear'], 'C': [0.1,1,10]}]
        parameters_svr_rbf = [{'kernel':['rbf'] , 'C': [50,75,100],'gamma': [0.001,0.01,0.1] }]
        svr_model_rbf = GridSearchCV(svreg, parameters_svr_rbf, cv=5, verbose=True, refit = True)
#Without Feature Selection      
        svr_model_rbf.fit(xtrain_scal,ytrain)
        ypred_svr_g_sfs = svr_model_rbf.predict(xtest_scal)
        RMSE_svr_g_sfs = np.sqrt(mean_squared_error(ytest,ypred_svr_g_sfs))
        MSE_svr_g_sfs = mean_squared_error(ytest,ypred_svr_g_sfs)
        MAE_svr_g_sfs = mean_absolute_error(ytest,ypred_svr_g_sfs)
        GS_NoFS = [RMSE_svr_g_sfs, MSE_svr_g_sfs, MAE_svr_g_sfs]
#With Threshole 
        svr_model_rbf.fit(xtrain_scal_var,ytrain)
        ypred_svr_g_var = svr_model_rbf.predict(xtest_scal_var)
        RMSE_svr_g_var = np.sqrt(mean_squared_error(ytest,ypred_svr_g_var))
        MSE_svr_g_var = mean_squared_error(ytest,ypred_svr_g_var)
        MAE_svr_g_var = mean_absolute_error(ytest,ypred_svr_g_var)
        GS_Threshole = [RMSE_svr_g_var, MSE_svr_g_var, MAE_svr_g_var]
#With Lasso      
        svr_model_rbf.fit(xtrain_scal_lasso,ytrain)
        ypred_svr_g_lasso = svr_model_rbf.predict(xtest_scal_lasso)
        RMSE_svr_g_lasso = np.sqrt(mean_squared_error(ytest,ypred_svr_g_lasso))
        MSE_svr_g_lasso = mean_squared_error(ytest,ypred_svr_g_lasso)
        MAE_svr_g_lasso = mean_absolute_error(ytest,ypred_svr_g_lasso)
        GS_Lasso = [RMSE_svr_g_lasso, MSE_svr_g_lasso, MAE_svr_g_lasso]

        GS_Stats = [GS_NoFS, GS_Threshole, GS_Lasso]
        return ypred_svr_g_sfs, ypred_svr_g_var, ypred_svr_g_lasso, GS_Stats


def KNN_Lasso (x_nodata_scal,lasso_cols,xtrain_scal_lasso,ytrain, age_no_data, df):
        #KNN Parameters    
        neigh = KNeighborsRegressor(weights = "distance")
        parameters_k = np.arange(2,41,2)
        parameters_knn = [{'n_neighbors': parameters_k}]
        knn_model = GridSearchCV(neigh, parameters_knn, refit = True, cv=5, verbose=True)
    
        x_sindatos_scal_lasso = x_nodata_scal[:, lasso_cols]
        knn_model.fit(xtrain_scal_lasso, ytrain)
        predicted_age_knn_lasso = knn_model.predict(x_sindatos_scal_lasso)
        
        lavf_edades_knn = age_no_data.copy() 
        lavf_edades_knn['victima_edad'] = predicted_age_knn_lasso

        lavf_knn = pd.concat([df,lavf_edades_knn], axis = 0,  join = "outer" , ignore_index = False)
        lavf_knn['victima_edad'] = lavf_knn['victima_edad'].round()
        lavf_knn.loc[(lavf_knn['victima_genero'] == "Trans"), 'victima_genero'] = "Transgénero"
        df_knn = lavf_knn
        
        return df_knn

def SVR_G_Threshole (x_nodata_scal, threshold_cols, xtrain_scal_var, ytrain, age_no_data, df):
        #Gauss Parameters
        parameters_svr_rbf = [{'kernel':['rbf'] , 'C': [50,75,100],'gamma': [0.001,0.01,0.1] }]
        svr_model_rbf = GridSearchCV(svreg, parameters_svr_rbf, cv=5, verbose=True, refit = True)

        x_sindatos_scal_var = x_nodata_scal[:, threshold_cols]
        data_size = len(x_sindatos_scal_var)
        x_sindatos_scal_var = x_sindatos_scal_var.reshape(data_size,-1)
        svr_model_rbf.fit(xtrain_scal_var,ytrain) #Fit done while training
        predicted_age_svr_threshole = svr_model_rbf.predict(x_sindatos_scal_var)

        lavf_edades_svrg = age_no_data.copy() 
        lavf_edades_svrg['victima_edad'] = predicted_age_svr_threshole
        lavf_svrg = pd.concat([df,lavf_edades_svrg], axis = 0,  join = "outer" , ignore_index = False)
        lavf_svrg['victima_edad'] = lavf_svrg['victima_edad'].round()
        lavf_svrg.loc[(lavf_svrg['victima_genero'] == "Trans"), 'victima_genero'] = "Transgénero"
        victima_edades_svrg = lavf_svrg.iloc[:,4]
        df_svrg = lavf_svrg
        return df_svrg


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
from sklearn import preprocessing    
    
    
    
def FSelection2(lavf_preselectionC):
        x2 = lavf_preselectionC.drop(['llamado_derivacion'], axis=1)
        y2 = lavf_preselectionC.iloc[:,11]
#TRAIN TEST SPLIT
        xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x2, y2, test_size=0.3, random_state=4)
        scaler = preprocessing.StandardScaler().fit(xtrain2)
        xtrain_scal2 = scaler.transform(xtrain2)
        xtest_scal2 = scaler.transform(xtest2)
#THRESHOLE
        thresh = 0.5
        xtrain_scal_red_var = xtrain2.iloc[:,(np.std(xtrain2)>np.quantile(np.std(xtrain2), thresh)).values]
        xtest_scal_red_var = xtest2.iloc[:,(np.std(xtrain2)>np.quantile(np.std(xtrain2), thresh)).values]
        x_threshold = x2.iloc[:,(np.std(xtrain2)>np.quantile(np.std(xtrain2), thresh)).values]
        #x_threshold.columns
#LASSO 
        lasso_featsel = Lasso(alpha = 0.01)
        lasso_featsel.fit(xtrain_scal2,ytrain2)

        Features = int(np.count_nonzero(lasso_featsel.coef_))

        lasso_cols = lasso_featsel.coef_ != 0
        x_lasso = x2.iloc[:,lasso_cols]
        xtest_scal_lasso2 = xtest_scal2[:, lasso_cols]
#RFE
        rfe = svm.SVC(C=10, kernel="linear")
        rfecv = RFECV(estimator=rfe, step=1, cv=5, scoring='accuracy')
        rfecv.fit(xtrain_scal2, ytrain2)
        x_rfe = x2.iloc[:,rfecv.support_] #columns
        xtrain_scal_lasso2 = xtrain_scal2[:, lasso_cols]
        xtrain_scal_rfe2 = xtrain_scal2[:, rfecv.support_]

        return xtrain_scal_lasso2, xtrain_scal_rfe2, xtest_scal_lasso2, ytrain2, ytest2
    
    
def KNN_Lasso2(xtrain_scal_lasso2, xtest_scal_lasso2, ytrain2, ytest2):

        parameters = {'n_neighbors':[1, 5, 10, 25, 50]}
        knn = KNeighborsClassifier()
        knn_cl = GridSearchCV(knn, param_grid = parameters, refit = True ,cv = 5)
        knn_cl.fit(xtrain_scal_lasso2,ytrain2.ravel())
        
        ypred_knn_train_l_cl = knn_cl.predict(xtrain_scal_lasso2) 
        knn_score_train_l_cl = accuracy_score(ytrain2,ypred_knn_train_l_cl)
        
        ypred_knn_test_l_cl = knn_cl.predict(xtest_scal_lasso2) 
        knn_score_test_l_cl = accuracy_score(ytest2,ypred_knn_test_l_cl)
        
        yproba_knn_test_l_cl = knn_cl.predict_proba(xtest_scal_lasso2)
        fpr1, tpr1, thresholds = roc_curve(ytest2.astype('int'), yproba_knn_test_l_cl[:,1], drop_intermediate = False)
        auc_knn_l_cl = metrics.auc(fpr1, tpr1)
        
        return fpr1, tpr1, auc_knn_l_cl, ypred_knn_test_l_cl

def SVM_Lasso(xtrain_scal_lasso2, xtest_scal_lasso2, ytrain2, ytest2):
        parameters = {'C':[1,10,25]}
        svc = svm.SVC(kernel='linear',probability = True)
        svm_cv = GridSearchCV(svc, param_grid = parameters, refit = True, cv = 5)
        svm_cv.fit(xtrain_scal_lasso2,ytrain2.ravel())
        ypred_svm_train = svm_cv.predict(xtrain_scal_lasso2) 
        svm_score_train = accuracy_score(ytrain2,ypred_svm_train)
        ypred_svm_test = svm_cv.predict(xtest_scal_lasso2) 
        svm_score_test = accuracy_score(ytest2,ypred_svm_test)
        yproba_svm_test_l = svm_cv.predict_proba(xtest_scal_lasso2)
        fpr2, tpr2, thresholds = roc_curve(ytest2.astype('int'), yproba_svm_test_l[:,1], drop_intermediate = False)
        auc_svm_l = metrics.auc(fpr2, tpr2)
        
        return fpr2, tpr2, auc_svm_l, ypred_svm_test
        
        
        