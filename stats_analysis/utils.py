# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:24:08 2022

@author: Pedro Dubiela
"""

from   optbinning       import BinningProcess
from   scipy.stats      import chi2_contingency
import numpy  as np
import pandas as pd


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
        
    # special
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
                status = f"It has special: {special}"
        
                df_without_special = df_aux[~df_aux[var_name].isin(special)]
                df_with_special    = df_aux[df_aux[var_name].isin(special)]
                
                is_possible = df_without_special[target_variable].unique().shape[0] > 1
                
                if is_possible:
                
                    status += " - The binning is possible"
                    
                    # Binning
                    categorical_var = [var_name] if var_name in categorical_variables else None
                    result  = binning(df_without_special, target_variable, categorical_var)
                    #summary = result[0]
                    df      = result[1]
                    
                else:
                    
                    status += " - The binning isn't possible"
                    
                    df = df_without_special
                
                # Concat
                df = pd.concat([df_with_special, df])
                
            else:
                status = f"It hasn't special: {special}"
                
                is_possible = df_aux[target_variable].unique().shape[0] > 1
                
                if is_possible:
                    
                    status += " - The binning is possible"
                
                    # Binning
                    categorical_var = [var_name] if var_name in categorical_variables else None
                    result  = binning(df_aux, target_variable, categorical_var)
                    #summary = result[0]
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
    min_bin_size          = 0.00007
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