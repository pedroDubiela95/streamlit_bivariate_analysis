

# Imports
from   stats_analysis import utils as u
import pandas                      as pd
import streamlit                   as st


################################# Sidebar #####################################

#st.sidebar.header('df e Variáveis')
# Load
st.sidebar.markdown("""# Variables""")
file                  = st.sidebar.file_uploader("## Upload your file", type={"csv"})
ready = False

# Columns
if file is not None:
    df = pd.read_csv(file)
    st.markdown("""Resumo dos Dados""")

    # Columns
    all_options_cols = df.columns.to_list() + ['All']
    
    # deixar tem que tratar o all o nonme
    numericals = st.sidebar.multiselect('Numerical Vars', all_options_cols)
    categoricals = list(set(all_options_cols) - set(numericals))
    
    categoricals = st.sidebar.multiselect('Categorical Vars', categoricals) 
    #numericals = st.sidebar.radio("Regularização:", categoricals)

    there_is_special_vars = st.sidebar.radio("Is there Special Vars:", ('No', 'Yes'))

    # Load Special
    special_var = None
    if there_is_special_vars == 'Yes':
        path_special_var = st.sidebar.file_uploader("## Upload you Sp", type={"csv"})
        if path_special_var is not None:
            special_var = pd.read_csv(path_special_var, sep = ",")
        
    
    ready = st.sidebar.button("Executar")
###############################################################################

c1 = st.container()
tab1, tab2, tab3 = st.tabs(['Dataset Sample', 'Describe', 'NaN Quantity'])
c2 = st.container()

# Texto inicial

with c1:
    st.markdown(""" 
                # DUB_analysis
                ## Analisador de variáveis
                Esta aplicação web tem por objetivo realizar uma análise de dados 
                detalhada em um df.
                """)

if file is not None:
    
    with tab1:
        st.write("Amostra", df.head())
    
    with tab2:
        st.write("Describe", df.describe())
    
    with tab3:
        nans = df.isna().sum().to_frame(name = "NaN Quantity")
        st.write("NaN", nans)
    


# Definindo as numéricas
numerical_variables = ['Col1','Col2'] 
              
# Definindo as categóricas (inclui numérica discreta)
categorical_variables = ['Col3', 'Col4', 'Cat']

# Definindo target
target_variable = 'flag'


not_binning = ['Col3', 'Col4']
remove_na = False
not_binning = []
l1          = 0.11
l2          = 0.30

#df = pd.read_csv("./source/file_test.csv", sep = ',')
#special_var = pd.read_csv("./source/variables_special.csv", sep = ",")

if file is not None:
    
    if there_is_special_vars == 'No':
    
        df_res = u.bivariate_analysis(
            df                    = df, 
            numerical_variables   = numerical_variables, 
            categorical_variables = categorical_variables, 
            target_variable       = target_variable, 
            remove_na             = remove_na,
            not_binning           = not_binning,
            l1                    = l1,
            l2                    = l2,
            special_var           = special_var)
    else:
        if special_var is not None:
            
            df_res = u.bivariate_analysis(
                df                    = df, 
                numerical_variables   = numerical_variables, 
                categorical_variables = categorical_variables, 
                target_variable       = target_variable, 
                remove_na             = remove_na,
                not_binning           = not_binning,
                l1                    = l1,
                l2                    = l2,
                special_var           = special_var)


if ready:
    with c2:
        
        if special_var is not None:
            st.markdown("Special Var Table")
            st.write(special_var)
        
        st.markdown("Bivariate Analysis")
        st.write(df_res)


#df_res.to_excel('./bivariate_analysis.xlsx', index=False) 
