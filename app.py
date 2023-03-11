# Imports
from   stats_analysis import utils as u
import pandas                      as pd
import streamlit                   as st


################################# Sidebar #####################################

#st.sidebar.header('df e Variáveis')
# Load
st.sidebar.markdown("""# Variables""")
file  = st.sidebar.file_uploader("## Upload your file", type={"csv"})
ready = False

# Columns
if file is not None:
    df = pd.read_csv(file, sep = ';', index_col = 0)
    st.markdown("""Resumo dos Dados""")

    # Columns
    all_options_cols = df.columns.to_list() + ['All']
    
    # Numericals
    numericals = st.sidebar.multiselect('Numerical Vars', all_options_cols)
    all_options_cols = list(set(all_options_cols) - set(numericals))
    
    # Categoricals
    categoricals = st.sidebar.multiselect('Categorical Vars', all_options_cols)
    all_options_cols = list(set(all_options_cols) - set(categoricals))
    
    # Target
    target = st.sidebar.selectbox('Target Var', df.columns.to_list()) 
    all_options_cols = list(set(all_options_cols) - set(target))
    
    if 'All' in numericals:
        numericals = ['All']
        categoricals = []

    if 'All' in categoricals:
        categoricals = ['All']
        numericals = []
    
    ready = st.sidebar.button("Executar")
###############################################################################

c1 = st.container()
tab1, tab2, tab3 = st.tabs(['Dataset Sample', 'Describe', 'NaN Quantity'])
c2 = st.container()

# Texto inicial

with c1:
    st.markdown(""" 
                # Analysis
                ## Analisador de variáveis
                Esta aplicação web tem por objetivo realizar uma análise de dados 
                detalhada em um df.
                """)

if file is not None:
    
    with tab1:
        st.write("Amostra", df.head().style.format(precision=2))
    
    with tab2:
        st.write("Describe", df.describe().style.format(precision=2))
    
    with tab3:
        nans = df.isna().sum().to_frame(name = "NaN Quantity")
        st.write("NaN", nans)
    

if file is not None:


    # Definindo as numéricas
    numerical_variables = numericals
                  
    # Definindo as categóricas (inclui numérica discreta)
    categorical_variables = categoricals
    
    # Definindo target
    target_variable = target
    
    df[numericals] = df[numericals].astype('float64')
    
    if ready:
        with c2:
            
            for var in numericals:

                st.markdown(var)
                res = u.exploratory_data_analysis_numerical(df, var)
                st.write(res.style.format(precision=2))

