from stats_analysis import utils as u
import numpy        as np
import pandas       as pd



def test_keep_only_numbers():
    
    df     = pd.DataFrame(
        {'Column A':['123.456.789-10', '123456', '1!2@3#4$5%6¨7&8*9(10)[]11sd']})
    
    df_res = u.keep_only_numbers(df, col_name='Column A')
    
    c1     =  df_res.loc[0,:].values[0] ==  '12345678910'
    c2     =  df_res.loc[1,:].values[0] ==  '123456'
    c3     =  df_res.loc[2,:].values[0] ==  '1234567891011'
    
    assert np.all([c1, c2, c3]) == True
    

def test_fix_strings():
    
    word = '    São Paulo, Segunda-feira, Paraná   '    
    c1 = u.fix_strings(word) == 'SAO PAULO, SEGUNDA-FEIRA, PARANA'
    assert c1 == True


def params_bivariate_analysis():
    
    df                    = pd.read_csv('./source/file_test.csv')
    numerical_variables   = ['var_B', 'var_D']
    categorical_variables = ['var_A', 'var_C']
    target_variable       = 'flag'
    not_binning           = ['var_A', 'var_B', 'var_C', 'var_D']
    special_var = pd.read_csv('./source/variables_special.csv')
    
    return (
        df, numerical_variables, categorical_variables, 
        target_variable, not_binning, special_var
        )

    
def test_bivariate_analysis1():
        (df, numerical_variables, categorical_variables, 
        target_variable, not_binning, special_var) = params_bivariate_analysis()
        
        # 1
        # Vai tentar binnar todo mundo
        # Vai manter os missing (trocando NaN por "Missing Values") em todas
        # Não vai realocar specials vars (não utiliza o excel)
        res = u.bivariate_analysis(df, numerical_variables, categorical_variables, target_variable)

        df1 = pd.read_excel('./source/data_tests/result1.xlsx', index_col=0)
        df1 = df1[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        res = res[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        r = df1 == res
        
        assert r.all().all() == True
        
        
def test_bivariate_analysis2():
        (df, numerical_variables, categorical_variables, 
        target_variable, not_binning, special_var) = params_bivariate_analysis()
        
        # 2
        # Nao vai binnar ninguém
        # Vai manter os missing (trocando NaN por "Missing Values") em todas
        # Não vai realocar specials vars (não utiliza o excel)
        res = u.bivariate_analysis(df, numerical_variables, categorical_variables, target_variable, not_binning=not_binning)

        df2 = pd.read_excel('./source/data_tests/result2.xlsx', index_col=0)
        df2 = df2[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        res = res[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        r = df2 == res
        
        assert r.all().all() == True
        
        
def test_bivariate_analysis3():
        (df, numerical_variables, categorical_variables, 
        target_variable, not_binning, special_var) = params_bivariate_analysis()
        
        # 3
        # Vai tentar binnar todo mundo
        # Vai retirar linhas com  missing em todas
        # Não vai realocar specials vars (não utiliza o excel)
        res = u.bivariate_analysis(df, numerical_variables, categorical_variables, 
                                   target_variable, remove_na=True)

        df3 = pd.read_excel('./source/data_tests/result3.xlsx', index_col=0)
        df3 = df3[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        res = res[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        r = df3 == res
        
        assert r.all().all() == True
        
        
def test_bivariate_analysis4():
        (df, numerical_variables, categorical_variables, 
        target_variable, not_binning, special_var) = params_bivariate_analysis()
        
        # 4
        # Nao vai binnar ninguém
        # Vai retirar linhas com  missing em todas
        # Não vai realocar specials vars (não utiliza o excel)
        res = u.bivariate_analysis(df, numerical_variables, categorical_variables, 
                                   target_variable, not_binning=not_binning, remove_na=True)

        df4 = pd.read_excel('./source/data_tests/result4.xlsx', index_col=0)
        df4 = df4[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        res = res[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        r = df4 == res
        
        assert r.all().all() == True
        
        
def test_bivariate_analysis5():
        (df, numerical_variables, categorical_variables, 
        target_variable, not_binning, special_var) = params_bivariate_analysis()
        
        # 5
        # Vai tentar binnar todo mundo
        # Vai manter os missing (trocando NaN por "Missing Values") em todas
        # Vai realocar specials vars (utiliza o excel)
        #
        res = u.bivariate_analysis(df, numerical_variables, categorical_variables, 
                                   target_variable, special_var = special_var)

        df5 = pd.read_excel('./source/data_tests/result5.xlsx', index_col=0)
        df5 = df5[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        res = res[['Indicador', 'Categoria', 'Nao_evento', 'Evento', 'Total']]
        r = df5 == res
        
        assert r.all().all() == True
   






    

    



    
    
    
    
    
    
    
    