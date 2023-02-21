# -*- coding: utf-8 -*-
"""
List all functions:
 
 1. binning
 2. calculate_CramersV
 3. create_model
 4. fix_strings
 5. keep_only_numbers
 6. processing_result

"""

from   optbinning              import BinningProcess
from   scipy.stats             import chi2_contingency
from   sklearn.preprocessing   import OneHotEncoder
from   sklearn.model_selection import train_test_split
from   sklearn                 import tree 
from   unicodedata             import normalize
import numpy                   as np
import pandas                  as pd
import pickle
import random
import re


def bivariate_analysis(df, 
                       numerical_variables, 
                       categorical_variables, 
                       target_variable, 
                       special_var = None, 
                       not_binning = [],
                       l1          = 0.11,
                       l2          = 0.30,
                       remove_na   = False):

    # Filter
    df = df[numerical_variables + categorical_variables + [target_variable]]
    
    
    # Forced Tipying
    if len(numerical_variables) !=0 :
            df[numerical_variables] = df[numerical_variables].astype(float)
        
    # If it must keep NaN or not
    if remove_na:
        df.dropna(inplace   = True)
        df.reset_index(drop = True, inplace = True)
    
    else:
        df.fillna('Missing Value', inplace = True)
        

    cols = df.drop([target_variable], axis = 1 ).columns
    
    # Result
    df_res = pd.DataFrame()
    df_original = df.copy()

    # For each variable in df (except target_variable)
    for var_name in cols:
        
        print(var_name)
            
        # 1 - Binning:-------------------------------------------------------------
        df_aux = df_original[[target_variable, var_name]]
        
        
        if np.any(special_var != None):
            special = special_var.loc[special_var['variable_name'] == var_name, 
                                       ['value1', 'value2', 'value3']].values[0]
        
            special = special.tolist() + ['Missing Value']
        
        else:
            special = ['Missing Value']
            
        has_special = np.any(df_aux[var_name].isin(special))   
        
                
        if var_name not in not_binning:
       
            if has_special:
                
                status = f"It has Special Values!"
    
                c = df_aux[var_name].astype(str).isin(special)
                
                df_without_special = df_aux[~c].reset_index(drop = True)
                df_with_special    = df_aux[c].reset_index(drop = True)
                
                is_possible = df_without_special[target_variable].unique().shape[0] > 1
                
                if is_possible:
                
                    status += " - The binning is possible"
                    
                    # Binning
                    categorical_var = [var_name] if var_name in categorical_variables else None
                    result  = binning(df_without_special, target_variable, categorical_var)
                    df      = result[1]
                    
                else:
                    
                    status += " - The binning isn't possible"
                    
                    df = df_without_special
                
                # Concat
                df = pd.concat([df_with_special, df])
                
            else:
                status = "It hasn't Special Values"
                
                is_possible = df_aux[target_variable].unique().shape[0] > 1
                
                if is_possible:
                    
                    status += " - The binning is possible"
                
                    # Binning
                    categorical_var = [var_name] if var_name in categorical_variables else None
                    result  = binning(df_aux, target_variable, categorical_var)
                    df      = result[1]
                
                else:
                    
                    status += " - The binning isn't possible"
                    
                    df = df_aux     
           
        else:
            status = "For this variable, we chose not to bin"
            df = df_aux
                    
        
        # 2 - Total, Event and Non-event ------------------------------------------
        total = df.groupby([var_name])[target_variable].count()
        event = df.groupby([var_name])[target_variable].sum()
        non_event = total - event
    
        df_res1 = pd.concat([total, event, non_event], axis=1).reset_index()
        df_res1.columns = ['Categoria' ,'Total', 'Evento', 'Nao_evento']
        df_res1["Indicador"] = var_name
        df_res1["Binning_Status"] = status
        
        # 3 - Cramers'V -----------------------------------------------------------
        var_x = var_name
        var_y = target_variable
        
        V = calculate_CramersV(var_name_x = var_x, 
                               var_name_y = var_y,
                               df = df)
    
        df_res1["Cramer's V"] = V
        df_res = pd.concat([df_res, df_res1])
        #--------------------------------------------------------------------------
    


    # Create metrics 2
    df_res['% Resposta'] = (df_res['Evento'] / (df_res['Total']))*100
    qtd_growers = df_res.groupby(['Indicador']).agg({'Total':'sum'})
    df_res['% Categoria'] = (df_res['Total'] / qtd_growers.iloc[0,0])*100
    
    
    # Classifying
    df_res['Discriminância'] = df_res["Cramer's V"].apply(
        lambda x : "Baixa" if x <= l1 else ("Média" if x <= l2 else "Alta"))
    
    # Create metrics 3
    df_res['Media da base'] = (sum(df_res['Evento']) / sum(df_res['Total']))
    
    # Sort
    df_res = df_res[['Indicador', 'Categoria',	'Nao_evento', 'Evento',	'Total', 
                     'Media da base', '% Resposta',	'% Categoria',	
                     "Cramer's V",	'Discriminância', 'Binning_Status']]
        
    df_res.reset_index(drop = True, inplace = True)
    
    return df_res


def binning(df, target_variable, categorical_variables = None):
    """
    This function performs a binning optimization for the variables contained 
    in the given dataframe (except for a target variable)
    
    See http://gnpalencia.org/optbinning/binning_process.html                        
    
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with all the variables that will be binned, including the 
        target variable.
    target_variable : str
        The name of the target variable.
    categorical_variables : list
        A list of all numeric variables that should be considered categorical.
    Returns
    -------
    binning_process.summary() : pd.DataFrame
        A dataframe with a summary of each binned variable, for example:
        dtype, status, n_bis etc
    df_res : pd.DataFrame
        A dataframe with all variables binned.
        
    """
    

    # All variables in df
    variable_names = np.ndarray.tolist(df.columns.values) 

    # Remove target_variable
    variable_names = [item for item in variable_names if item != target_variable]
    
    min_n_bins            = 2
    max_n_bins            = 3 
    min_bin_size          = 0.001
    binning_fit_params    = {'min_bin_n_event':4}
    
    X = df[variable_names].values
    y = df[target_variable].values
    
    if categorical_variables != None:
        binning_process = BinningProcess(variable_names,
                                         categorical_variables = categorical_variables,
                                         max_n_bins            = max_n_bins , 
                                         min_bin_size          = min_bin_size , 
                                         min_n_bins            = min_n_bins, 
                                         binning_fit_params    = binning_fit_params)
    else:
        binning_process = BinningProcess(variable_names,
                                         max_n_bins            = max_n_bins , 
                                         min_bin_size          = min_bin_size , 
                                         min_n_bins            = min_n_bins, 
                                         binning_fit_params    = binning_fit_params)
        
    binning_process.fit(X, y)  
    X_transform = binning_process.transform(X, metric="bins") 
    
    df_res  = pd.DataFrame(data=X_transform, columns=variable_names)
    df_res[target_variable] = y
    
    binning_process.information()
    
    return (binning_process.summary(), df_res)


def calculate_CramersV(var_name_x, var_name_y, df):
    """
    This function calculates the degree of association between two categorical 
    variables contained in the dataframe. The metric used is Cramer's V.
    
    See 
        FÁVERO, Luiz Paulo Lopes e BELFIORE, Patrícia Prado. Manual de análise 
        de dados: estatística e modelagem multivariada com excel, SPSS e stata.
        Rio de Janeiro: Elsevier.
    Parameters
    ----------
    var_name_x : str
        The X variable name.
    var_name_y : str
        The Y variable name..
    df : pd.DataFrame
        A dataframe with X and Y variables.
    Returns
    -------
    V : float
        Cramer's V coefficient of the relationship between X and Y.
    """
 
    X = df[var_name_x]
    Y = df[var_name_y]
    df_cross_tab = pd.crosstab(index  = X, columns = Y, margins = True)
    
    # 2) Calculate Chi^2
    result = chi2_contingency(observed = df_cross_tab.iloc[:-1, :-1])
    x2 = result[0]
    
    # 3) Calculate Cramer's V
    q = np.min(df_cross_tab.iloc[:-1, :-1].shape)
    n = df_cross_tab.loc["All", "All"]
    V = np.sqrt(x2/(n*(q-1))) if x2 != 0 else 0
    
    return V


def create_model(**kwargs):
    """
    This function receives a dataframe, along with the names of its numerical, 
    categorical and target variables. It also receives the hyperparameters of 
    the algorithm and builds a decision tree model.

    Parameters
    ----------
    **kwargs :
        df : pd.DataFrame
            A dataset
        numerical: list
            numerical variables's name
        categorical: lsit
            numerical variables's name
        target: list (it must has len == 1)
            target variable's name
        test_size: float
            size of test dataset for training
        min_samples_leaf: float
            It is the minimum number of samples that must exist on each leaf
        max_depth: float
            max depth of tree
        min_samples_split: float 
            is the minimum number of samples needed to split an internal node
        model_name.
            model name

    Returns
    -------
    clf : DecisionTreeClassifier
        Sci kit learning decision tree model.

    """

    # Parameters:
    df                = kwargs.get('df')
    numerical         = kwargs.get('numerical')
    categorical       = kwargs.get('categorical')
    target            = kwargs.get('target')
    test_size         = kwargs.get('test_size')
    min_samples_leaf  = kwargs.get('min_samples_leaf')
    max_depth         = kwargs.get('max_depth')
    min_samples_split = kwargs.get('min_samples_split')
    model_name        = kwargs.get('model_name')
    
    if target == None: raise Exception("Sorry, target is mandatory")
    
    if (numerical != None) and (categorical != None):
        df = df[numerical + categorical + target]
    elif categorical == None: 
        df = df[numerical + target]
    else: 
        df = df[categorical + target]    
    
    # Variable Definition
    y = df.pop(target[0])
    X = df

    # Split train and test
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, 
        y, 
        range(len(y)), 
        test_size    = test_size,
        random_state = 1)

    # If there are categorical variables
    if categorical != None :

        # One-hot Encoding
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(X_train[categorical].astype(np.str))
        
        # Save
        with open(f'./models_results/enc_{model_name}.pkl', 'wb') as f:
            pickle.dump(enc, f)
    
        del enc    
    
        # Load saved model
        with open(f'./models_results/enc_{model_name}.pkl', 'rb') as f:
            enc = pickle.load(f)
        
     
        # Apply transform
        colnames = enc.get_feature_names_out()
        X_train_categorical = enc.transform(X_train[categorical].astype(np.str)).toarray()
        X_test_categorical  = enc.transform(X_test[categorical].astype(np.str)).toarray()
     
        # From np.array() to pd.DataFrame()
        X_train_categorical = pd.DataFrame(columns=colnames, data =  X_train_categorical)
        X_test_categorical = pd.DataFrame(columns=colnames, data =  X_test_categorical)
        
        # Drop categorical variables without encoding
        X_train.drop(categorical, axis = 1, inplace = True)
        X_test.drop(categorical, axis = 1, inplace = True)
        
        # Merge
        X_train = pd.concat([X_train.reset_index(drop = True), X_train_categorical], axis=1)
        X_test = pd.concat([X_test.reset_index(drop = True), X_test_categorical], axis=1)
    
    
    #Building classifier
    clf = tree.DecisionTreeClassifier(min_samples_leaf  = min_samples_leaf, 
                                      random_state      = 1, 
                                      max_depth         = max_depth,
                                      min_samples_split = min_samples_split)
    
    # Train
    clf.fit(X_train, y_train)
    
    # Save
    with open(f'./models_results/{model_name}.pkl', 'wb') as f:
        pickle.dump(clf, f)
        
    del clf

    # Load saved model
    with open(f'./models_results/{model_name}.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    return clf


def fix_strings(text):
    """
    This function takes a string and removes the accents, puts it in upper case 
    and trims the white spaces.

    Parameters
    ----------
    text : str
        The string that will be edited.

    Returns
    -------
    text : str
        The new edited string.

    """
    text = normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.strip().upper()
    return text

def keep_only_numbers(df, col_name = 'cat_grower_document'):
    """
    Preprocesses the data of a given column. Keep only numbers.    

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe that has the colname.
    col_name : str, optional
        The colname that will be 'normalized'. 
        The default is 'cat_grower_document'.

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame with the preprocessed colname.

    """
    df           = df[~df[col_name].isna()]
    df[col_name] = df[col_name].astype('str')
    df[col_name] = df[col_name].apply(lambda x : ''.join(re.findall(r'\d+', re.sub('\.0$', '', x))))
    df[col_name] = df[col_name].astype('str')   
    return df


def processing_result(**kargs):
    """
    This function performs the processing of a list of dataframes in such a 
    way as to remove the repeated col_name_x values, selecting those that have 
    the maximum value of col_name_y. 
    For example, consider col_name_x being the grower's document column and the 
    col_name_y column being a discrete numeric variable that varies between 0 
    and 1, in this case, this function will select the grower's document that 
    has the value 1, if there is more than one or none, the function draws 
    randomly.
    
    Parameters
    ----------
    **kargs :
        It must be a list with all the pd.DataFrame()
 
        df : pd.DataFrame
            DESCRIPTION.
            
    Returns
    -------
    df_res : pd.DataFrame
        Processed data frae.

    """
    
    list_dfs = kargs.get('list_dfs')
    opt      = kargs.get('opt')
    
    # It converts from pd to np
    for i, item in enumerate(list_dfs):
        
        # Fix string (replace , with .)
        for j in range(item.shape[1]):
            
            try:
                item.iloc[:,j] = item.iloc[:,j].apply(lambda x : x.replace(',', '.'))
            except:
                pass
        
        # str to float
        if i == 0:
            arr = item.values.astype(float)
        else:
            values = item.values.astype(float)
            arr = np.append(arr, values, axis = 0)
           
    # arr[:, 0] - grower document
    # arr[:, 1] - flag_answer
    # arr[:, 2] - base
            
    # Get only distinct documents
    distinct_x  = np.unique(arr[:,0])
    
    for i, x in enumerate(distinct_x):
        
        # It groups by x
        arr_grouped_by_x = arr[arr[:,0] == x]
        
        # It gets max value of col_name_y
        max_y = arr_grouped_by_x[: , 1].max()
        
        # Get only rows which owns max value of y
        arr_grouped_by_res = arr_grouped_by_x[arr_grouped_by_x[:, 1] == max_y]
        
        
        #---------------------------------------------------------------------#
        # It performs the random selection for all producers, either 0 or 1.
        if opt == 1:
            
            # It performs the random selection
            if arr_grouped_by_res.shape[0] > 1:
                idx = random.randint(0, arr_grouped_by_res.shape[0] - 1)
                arr_grouped_by_res = arr_grouped_by_res[idx, : ]
        
        #---------------------------------------------------------------------#
        # If max(flag_answer) == 1, take the one with maximum base
        if opt == 2:

            # It gets the row with maximum base value
            if max_y == 1:
            
                max_base           = arr_grouped_by_res[:, 2].max()
                arr_grouped_by_res = arr_grouped_by_x[(arr_grouped_by_x[:, 1] == max_y ) & 
                                                      (arr_grouped_by_x[:, 2] == max_base)]
                
            else:
                # It performs the random selection
                if arr_grouped_by_res.shape[0] > 1:
                    idx = random.randint(0, arr_grouped_by_res.shape[0] - 1)
                    arr_grouped_by_res = arr_grouped_by_res[idx, : ]
        #----------------------------------------------------------------------
                    
        # It performs random selection only for max(flag_answer) == 0, 
        # if max(flag_answer) == 1, it keeps duplicates with flag_answer == 1.
        if opt == 3:
            
            if max_y == 0:
                
                # It performs the random selection
                if arr_grouped_by_res.shape[0] > 1:
                    idx = random.randint(0, arr_grouped_by_res.shape[0] - 1)
                    arr_grouped_by_res = arr_grouped_by_res[idx, : ]
                    
        #----------------------------------------------------------------------
        
        try:
            arr_grouped_by_res.shape[1] != 3
        except:
            arr_grouped_by_res = arr_grouped_by_res.reshape(1, -1)
                
        if i == 0 :         
            arr_res  = arr_grouped_by_res
        else:
            arr_res   = np.append(arr_res, arr_grouped_by_res, axis = 0)
  
    return arr_res


