import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from dotenv import load_dotenv
import os

load_dotenv()

with st.sidebar:
    # flow control side bar
    Side_bar_ran = False

    file = st.file_uploader(label='upload csv or xlsx file',
                            type=['csv', 'xlsx'])
    if file is not None:
        if file.type == 'text/csv':
            statement_data = pd.read_csv(file, index_col=None)

        elif file.type == "application/vnd.openxmlformats-officedocument"\
                          ".spreadsheetml.sheet":
            statement_data = pd.read_excel(io=file, sheet_name='Transaction Details',
                                 skiprows=6, engine='openpyxl')
        else:
            print('not in the file type')

        statement_data.dropna(subset=['Category'], inplace=True)

        statement_data_json = statement_data.to_json()

        url = st.text_input('prediction model app link')
        pred_url = url + '/predict'

        if st.button(label='run prediction'):
            response = requests.post(pred_url, json=statement_data_json,
                                     timeout=60)
            st.write(f'response code: {response.status_code} ')
            statement_data['pred_category'] = response.json()['category']
            # flow control side bar
            Side_bar_ran = True
    st.write('Cold start from Cloud run might cause errors, rerun if runs into issue')
    st.write('Dollar Amounts have been changed for privacy')


def data_cleanup(df_statement_data: pd.DataFrame) -> pd.DataFrame:
    '''remove extra fields and mask actual spend'''
    df_stat = df_statement_data.copy()
    scale_factor = os.getenv('scaleFactor')

    keep_cols = ['Date', 'Description', 'City/State',
                 'Amount', 'Reference',
                 'Category', 'pred_category']
    df_stat = df_stat.loc[:, keep_cols]
    df_stat['Amount'] = df_stat['Amount'].astype(float)
    df_stat['Amount'] = (df_stat['Amount']
                           .apply(lambda x: x*float(scale_factor))
                           )
    df_stat['Category'] = df_stat['Category'].str.split('-').str[0]

    for col in ['Description', 'City/State']:
        df_stat[col] = df_stat[col].str.lower()
        df_stat[col] = (
            df_stat[col]
            .str.replace(r'\d+', '', regex=True)
            .str.replace('[^\w\s]', '', regex=True)
            .str.replace('\n', ' ', regex=True)
        )
        df_stat[col] = df_stat[col].str.strip()

    df_stat['Description']= (df_stat['Description']
                             .str.replace(r'\s+', ' ', regex=True))

    df_stat['Description'] = [
        x.replace(str(y), '') for x,y in
        zip(df_stat['Description'],
            df_stat['City/State'])
    ]

    return df_stat.loc[:, df_stat.columns !='City/State']


if Side_bar_ran:
    df_data = data_cleanup(df_statement_data=statement_data)

    # spend 5 by category
    cat1_spend = df_data.groupby("pred_category")["Amount"].sum()
    cat1_spend = cat1_spend[:6]
    cat1_labels = [f"{x} (${y:.2f})" for x, y in zip(cat1_spend.index, cat1_spend.values)]
    cat1_fig = px.pie(cat1_spend, values=cat1_spend.values, names=cat1_labels, title="Top 5 Category Spend Breakdown")
    cat1_fig.update_traces(textposition='inside', textinfo='label')
    st.plotly_chart(cat1_fig, use_container_width=True)
    st.write(f"total spend(plus rent): {df_data.Amount.sum()+2130}")

    # top 5 spend by place
    top_n_by_places = df_data.groupby('Description')['Amount'].sum().nlargest(10)
    top5_fig = px.bar(top_n_by_places, x=top_n_by_places.values,
                      y=top_n_by_places.index, orientation='h',
                      title="Top n Spend by Description")
    top5_fig.update_traces(texttemplate='%{x:.2s}', textposition='outside', 
                           text=top_n_by_places.index)
    top5_fig.update_layout(xaxis_title="Spend", yaxis_title="Places",
                           yaxis_tickangle=0, barmode='stack',
                           yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(top5_fig, use_container_width=False)
    st.dataframe(df_data)