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
sep = st.sidebar.radio(
        "## Choose file separator",
        key     = "visibility", 
        options = [",", ";"])


# Columns
if file is not None:
    
    if sep == ';':
        df = pd.read_csv(file, sep = sep, index_col = 0)
    else:
        df = pd.read_csv(file, sep = sep)
        
    st.markdown("""Data Summaries""")
    
    there_is_numericals   = st.sidebar.selectbox('Is there numerical variables ?', ['Yes', 'No'], index = 1)
    there_is_categoricals = st.sidebar.selectbox('Is there categoricals variables ?', ['Yes', 'No'], index = 1)
    
    there_is_numericals   = True if there_is_numericals == 'Yes' else False
    there_is_categoricals = True if there_is_categoricals == 'Yes' else False

    if there_is_numericals and there_is_categoricals :

        d = {}
        for col in df.columns:
            d[col] = st.sidebar.selectbox(col, ['Numerical', 'Categorical'], index = 1)
             
        numericals   = [key for key, value in d.items() if value == 'Numerical']
        categoricals = [key for key, value in d.items() if value == 'Categorical']
    
    if there_is_numericals and not there_is_categoricals:
        numericals   = df.columns.tolist()
        categoricals = []
        
    if there_is_categoricals and not there_is_numericals:
        categoricals     = df.columns.tolist()
        numericals       = []

    ready = st.sidebar.button("RUN")
###############################################################################

# Containers
c1 = st.container()
tab1, tab2 = st.tabs(['Dataset Sample', 'NaN Quantity'])
 
c2 = st.container()


# Texto inicial

with c1:
    st.markdown(""" 
                # Analysis
                ## Exploratory Data Analysis
                This web application aims to perform an exploratory analysis on a set of data stored in csv format.

                Owner: Pedro G. Dubiela | pedro.dubielabio@gmail.com | https://github.com/pedroDubiela95
                """)

if file is not None:
     
    with tab1:
        st.write("Sample", df.head(10).style.format(precision=2))
    
    with tab2:
        nans = df.isna().sum().to_frame(name = "NaN Quantity")
        st.write("NaN", nans)
    

if file is not None:

    if ready:
        
        with c2:
            tab3, tab4 = st.tabs(['Quantitative', 'Qualitative'])
            
            with tab3:
            
                if there_is_numericals:
                    df[numericals] = df[numericals].astype('float64')
                    for var in numericals:
                        st.markdown(var)
                        res = u.exploratory_data_analysis_numerical(df, var)
                        st.write(res.style.format(precision=2))
                    
            with tab4:
                
                if there_is_categoricals:
                    df[categoricals] = df[categoricals].astype('str')
                    for var in categoricals:
                        st.markdown(var)
                        #res = u.exploratory_data_analysis_cat(df, var)
                        #st.write(res.style.format(precision=2))

