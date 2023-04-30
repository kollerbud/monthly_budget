import streamlit as st
import pandas as pd
import requests


with st.sidebar:

    file = st.file_uploader(label='upload csv or xlsx file',
                            type=['csv', 'xlsx'])
    if file is not None:
        if file.type == 'text/csv':
            data = pd.read_csv(file, index_col=None)

        elif file.type == "application/vnd.openxmlformats-officedocument"\
                          ".spreadsheetml.sheet":
            data = pd.read_excel(io=file, sheet_name='Transaction Details',
                                 skiprows=6, engine='openpyxl')
        else:
            print('not in the file type')

        send_data = data.to_json()
        url = 'http://127.0.0.1:8080/predict'

        response = requests.get(url, json=send_data)
        st.write('running')
        st.write(response.json())
# sending "data" to cloud run

