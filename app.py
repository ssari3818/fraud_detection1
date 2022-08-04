import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import base64
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from tensorflow import keras

st.set_page_config(layout='wide')
st.sidebar.title('Fraud Information')
st.image("fraud.jpg", use_column_width=True)

html_temp = """
<div style="background-color:Green;padding:10px">
<h2 style="color:white;text-align:center;">Fraud Detection - Group - 5</h2>
</div><br>"""

st.markdown(html_temp,unsafe_allow_html=True)

model = pickle.load(open('random_forest', 'rb'))


time = st.sidebar.slider('Time', min_value=1000.000000, max_value=200000.0000, value=1000.0000,
                         step=1000.0500)
v1 = st.sidebar.slider('V1', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v2 = st.sidebar.slider('V2', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v3 = st.sidebar.slider('V3', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v4 = st.sidebar.slider('V4', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v5 = st.sidebar.slider('V5', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v6 = st.sidebar.slider('V6', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v7 = st.sidebar.slider('V7', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v8 = st.sidebar.slider('V8', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v9 = st.sidebar.slider('V9', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v10 = st.sidebar.slider('V10', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v11 = st.sidebar.slider('V11', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v12 = st.sidebar.slider('V12', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v13 = st.sidebar.slider('V13', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v14 = st.sidebar.slider('V14', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v15 = st.sidebar.slider('V15', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v16 = st.sidebar.slider('V16', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v17 = st.sidebar.slider('V17', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v18 = st.sidebar.slider('V18', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v19 = st.sidebar.slider('V19', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v20 = st.sidebar.slider('V20', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v21 = st.sidebar.slider('V21', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v22 = st.sidebar.slider('V22', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v23 = st.sidebar.slider('V23', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v24 = st.sidebar.slider('V24', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v25 = st.sidebar.slider('V25', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v26 = st.sidebar.slider('V26', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v27 = st.sidebar.slider('V27', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
v28 = st.sidebar.slider('V28', min_value=-100.0000, max_value=100.0000, value=0.0000, step=0.0500)
amount = st.sidebar.slider('Amount', min_value=0.0000, max_value=1000000.0000, value=100.0000,
                           step=200.0000)
amount_scaler = st.sidebar.slider('Amount_Scaler', min_value=0.0000, max_value=1000000.0000, value=100.0000,
                           step=200.0000)						   


coll_dict = {'Time':time, 'V1':v1, 'V2':v2, 'V3':v3, 'V4':v4, 'V5':v5, 'V6':v6, 'V7':v7, 'V8':v8, 'V9':v9, 'V10':v10, 'V11':v11, 'V12':v12, 'V13':v13,
         'V14':v14, 'V15':v15, 'V16':16,
         'V17':17, 'V18':18, 'V19':19, 'V20':20, 'V21':v21, 'V22':v22, 'V23':v23, 'V24':v24, 'V25':v25, 'V26':v26, 'V27':v27, 'V28':v28,
         'Amount':amount,"Amount_Scaler":amount_scaler}
columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
         'V14', 'V15', 'V16',
         'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
         'Amount']





df_coll = pd.DataFrame.from_dict([coll_dict])
user_inputs = pd.get_dummies(df_coll,drop_first=True).reindex(columns=columns, fill_value=0)


scalerfile = 'scalercim'
scaler = pickle.load(open(scalerfile, 'rb'))

user_inputs_transformed = scaler.transform(user_inputs)

prediction = model.predict(user_inputs_transformed)


html_temp = """
<div style="background-color:Black;padding:10px">
<h2 style="color:white;text-align:center;">Fraud Detection - Group - 5</h2>


</div><br>"""

st.table(df_coll)

st.subheader('Click PREDICT if configuration is OK')

if st.button('PREDICT'):
	if prediction[0]==0:
		st.success(prediction[0])
		st.success(f'Not Fraud :)')
	elif prediction[0]==1:
		st.warning(prediction[0])
		st.warning(f'Fraud :(')

x= st.sidebar.html_temp = """
<div style="background-color:Green;padding:10px">
<h2 style="color:white;text-align:center;">C9155 Mesut Furkan - D1430 Sami - D1467 Ahmet Emin - D1468 Şuayip - D1664 Fatih </h2>
</div><br>"""

st.sidebar.markdown(x,unsafe_allow_html=True)