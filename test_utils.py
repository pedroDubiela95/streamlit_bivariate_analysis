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
