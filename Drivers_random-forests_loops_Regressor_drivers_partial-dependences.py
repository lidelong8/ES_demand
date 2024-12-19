import time,re,random
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.interpolate import splev, splrep 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


import sys
#sys.path.append(r'C:\Users\User\AppData\Roaming\Python\Python39')

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
#from sklearn.inspection import plot_partial_dependence 
from sklearn.inspection import partial_dependence 
from sklearn.compose import ColumnTransformer
from collections import defaultdict

config = {
    "font.family":'serif', 
    "font.size": 14,
    'font.weight' : 'bold',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    "axes.unicode_minus": False
}
rcParams.update(config)
###########################
#['scale_slope','scale_slope_deficit','scale_slope_surplus','Sensitivity_slope','Sensitivity_slope_deficit','Sensitivity_slope_surplus']
th=0
#writer = pd.ExcelWriter(r'C:\Users\User\Desktop\Regressor_drivers_permute_decolinear_cn.xlsx',engine='openpyxl')
for name_sheet in [ 'drivers','AP-dominated','RE-dominated','Balanced']: # name_sheet = 'drivers' 'drivers' 'drivers',
    #name_sheet='scale_slope'
    print(name_sheet)
    th=th+1;
    df = pd.read_excel(r'D:\ES_demand\Data_process_rf_20240929.xls',sheet_name=name_sheet)

    columns=df.columns.values
    cls=pd.Index([1,1,1,1,1,1,1,2,2,2,2,2,2]) #.astype('object').T ['sex','age','education','income','interest','times','sizes','PM2.5','VC','GDP','POP','PRE','TEM']
    nums=pd.Index([0,1,1,1,1,1,1,1,1,1,1,1,1]) #['score','sex','age','education','income','interest','times','sizes','satisfaction','PM2.5','VC','GDP','POP','PRE','TEM']
    x_columns = df.columns.tolist()
    x_columns.remove(columns[0])
    X = df[x_columns]
    y = df[columns[0]]
    X.columns=x_columns
    x_columns=[name.replace('_','') for name in x_columns]
    x_columns=pd.Index(x_columns)
    numerical_columns = [columns[2],columns[3],columns[4],columns[5],columns[6],columns[7],columns[8],columns[9],columns[10],columns[11],columns[12],columns[13]]
    categorical_columns = [columns[1]]
    ########################
    start =time.time()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=0)

    selected_features=[0,1,2,3,4,5,6,7,8,9,10,11,12]
    typ=cls[selected_features]
    nums=nums[selected_features]
    selected_features_names = X.columns[selected_features]

    X_train_sel = X_train[selected_features_names]
    X_test_sel = X_test[selected_features_names]
    X_columns_sel=selected_features_names
    
    print(X_columns_sel)
    print(X_columns_sel.shape[0])

    ## preprocessing missing values
    #categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=0)
    categorical_encoder = OrdinalEncoder(handle_unknown="error")
    numerical_pipe = SimpleImputer(strategy="mean")
    preprocessing = ColumnTransformer([("cat", categorical_encoder, categorical_columns),("num", numerical_pipe, numerical_columns),],verbose_feature_names_out=False,)

    rf_sel =  Pipeline([("preprocess", preprocessing),("Regressor", RandomForestRegressor(bootstrap=True,n_estimators=200,n_jobs=-1,oob_score=True,random_state=0,min_samples_leaf=1)),])
    rf_sel.fit(X_train, y_train)
    print('Running time: %s Seconds'%( time.time()-start))

    # Error printing
    print("Baseline accuracy on train data with features removed:"f" {rf_sel.score(X_train, y_train):.4}") # calculate the gini scores
    print("Baseline accuracy on test data with features removed:"f" {rf_sel.score(X_test, y_test):.4}")  # calculate the gini scores
    ############
    ########################Importances  plot
    #####Permutation Importances   
    print('Permutation Importances (train set)...')  
    start =time.time()
    result =permutation_importance(rf_sel, X_train_sel, y_train, n_repeats=20, random_state=0, n_jobs=-1)
    print('Running time: %s Seconds'%( time.time()-start))
    importances=result.importances_mean 
    sorted_idx = importances.argsort() [::-1]
    ##
    txts=pd.Index(['Gender','Age','Edu.','Income','Interest','Frequency','Sizes','PM2.5','VC','GDP','POP','Precip.','Temp.'])
    #names=[i.capitalize() for i in X_columns_sel[sorted_idx]]
    names=['%s'%re.sub(r'(\s+)', r'\0$ $', i) for i in txts[sorted_idx]]
    #names=X_columns_sel[sorted_idx]
    permute_importances =pd.Series(importances[sorted_idx], index=names)

    #text (chr(97+i))
    if name_sheet.find('drivers')>=0:
      titl='All'
    else:
      titl=name_sheet
    #fig.savefig('C:/Users/User/Desktop/Regressor_drivers_sig_importances_decolinear_global_%s.jpg'%name,dpi=350,bbox_inches='tight')
    ########################Importances  plot    
    ########################    
    # Save key parameters
    #df_imp = pd.DataFrame()
    #df_imp['RI_'+ name_sheet] = list(result.importances_mean)
    #df_imp.index =X_columns_sel
    #df_imp.to_excel(excel_writer=writer, sheet_name=name_sheet)
 ########################
    print("Computing partial dependence plots...")
    start=time.time()
    sorted_idx = result.importances_mean.argsort() [::-1]
    lb=X_columns_sel[sorted_idx]
    im=result.importances_mean[sorted_idx]
    typc=typ[sorted_idx]
    nums=nums[sorted_idx]
    features =list([lb[0],lb[1],lb[2],lb[3],lb[4],lb[5],lb[6],lb[7],lb[8],lb[9],lb[10],lb[11],lb[12]])
    
     
    ### plot fiure
    sns.set_theme(style="ticks", palette="deep", font_scale = 1.1)
    fig = plt.figure(figsize=(20, 20)) #34 12  
    ax  = plt.subplot(5,3,(1,2))

    x1=[x+1 for x in range(len(typ)) if typc[x]==1]
    y1=[permute_importances[x] for x in range(len(typ)) if typc[x]==1]
    x2=[x+1 for x in range(len(typ)) if typc[x]==2]
    y2=[permute_importances[x] for x in range(len(typ)) if typc[x]==2]
    x3=[x+1 for x in range(len(typ)) if typc[x]==3]
    y3=[permute_importances[x] for x in range(len(typ)) if typc[x]==3]
    ax.bar(x1,y1, label="Socio-economic or personal varaibles",color="blue")
    ax.bar(x2,y2, label="Environmental varaibles",color="red")
    #ax.bar(x3,y3, label="Plant",color="green")
    ax.set_title(titl, fontsize=20, fontweight='bold',fontname='Arial')
    ax.set_ylabel("Variable Importance", fontsize=18, fontweight='bold',fontname='Arial')
    ax.set_xticks(range(1,len(typ)+1))
    ax.set_xticklabels(names,rotation=20, fontsize=18, fontweight='bold',fontname='Arial') 
    #ax.xaxis.set_tick_params(labelrotation=30) #sucess
    #ax.set_xticklabels(ax.get_xticklabels(),rotation=30) #ax.get_xticks() #sucess
    #ax.axvline(x=0.1, color="k", linestyle="--")
    #ax.axhline(y=0.1, color="k", linestyle="--")
    ax.legend(loc=0,title='')  #Drivers types
    #fig.text(0.22, 0.93,'Gini scores of 90%% train data: %.3f'%rf_sel.score(X_train_sel, y_train),fontsize=12, fontweight='bold',horizontalalignment='left')
    #fig.text(0.22, 0.91,'Gini scores of 10%% test data: %.3f'%rf_sel.score(X_test_sel, y_test),fontsize=12, fontweight='bold',horizontalalignment='left')
    count=0
    ax.text(-0.01, 1.05, ('(%s)'%chr(97+count)), fontsize=20, fontweight='bold', transform=ax.transAxes)
    
    count=0
    for i in features:
        count=count+1
        pdp = partial_dependence(rf_sel, X_train_sel, [i], kind="both",response_method='auto', method='brute', grid_resolution=50) # 除均值计算出每个样本的ice
        #The response_method parameter is ignored for regressors and must be 'auto'
        ax  = plt.subplot(5,3,count+2)

        plot_x = pd.Series(pdp['values'][0]).rename('x')
        plot_i = pdp['individual'] 
        plot_y = pdp['average'][0]


        plot_df = pd.DataFrame(columns=['x','y']) 
        for a in plot_i[0]:
            a2 = pd.Series(a)
            df_i = pd.concat([plot_x, a2.rename('y')], axis=1) 
            ##plot_df = plot_df.append(df_i)
            plot_df = pd.concat([plot_df,df_i],  ignore_index=True)

        if nums[count-1]==1:
          sns.lineplot(data=plot_df, x="x", y="y", color='k', linewidth = 1.5, linestyle='--', alpha=0.6) 
          #plt.plot(xnew, ynew, linewidth=2) 
          x_min = plot_x.min()-(plot_x.max()-plot_x.min())*0.1
          x_max = plot_x.max()+(plot_x.max()-plot_x.min())*0.1
        else:
          ax=sns.barplot(data=plot_df, x="x", y="y", color='b',ci=95,errcolor='k', errwidth=2, alpha=0.6) 
          ax.set_xticks([0,1])
          ax.set_xticklabels(['Male','Female'],rotation=0, fontsize=14, fontweight='bold',fontname='Arial') 
          x_min = -0.5
          x_max =1.5

        if (count+2)%3 == 1:
          plt.ylabel('Trade-off intensity', fontsize=18, fontweight='bold',fontname='Arial')
        else:
          plt.ylabel('', fontsize=18, fontweight='bold',fontname='Arial')
        plt.xlabel('%s'%names[count-1], fontsize=18, fontweight='bold',fontname='Arial') #$%s$
        plt.xlim(x_min,x_max)
        ax.text(-0.01, 1.05, f'({chr(97+count)})', fontsize=20, fontweight='bold', transform=ax.transAxes)
        #plt.text(0.01, 1.01, ('(%s)'%chr(97+count)), fontsize=16, fontweight='bold')
        #plt.ylim(0.4,0.6)




    fig.tight_layout() 
    fig.savefig(r'D:\ES_demand\Regressor_drivers_sig_Partial_dependence_decolinear_%s.jpg'%titl,dpi=350,bbox_inches='tight')

    print('Finished...'+name_sheet)

print('Running time: %s Seconds'%(time.time()-start))
print('All Finished!')
    ########################
