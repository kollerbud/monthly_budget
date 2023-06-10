import os
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from dotenv import load_dotenv
from query_bq import RecentSpendsFromBQ

load_dotenv()


df_statements = RecentSpendsFromBQ().build_query(
                cols=['Date', 'Description', 'City_State', 'Amount',
                      'Zip_Code', 'Category'],
                num_months=3
                )
# required columns
# 'Description', 'Amount', 'City/State', 'Zip Code', 'Category'
df_statements.rename(columns={'City_State': 'City/State',
                              'Zip_Code': 'Zip Code'},
                     inplace=True
                     )
# drop empty category, it is card payment
df_statements.dropna(subset=['Category'], inplace=True)


with st.sidebar:
    # flow control side bar
    Side_bar_ran = False

    statement_data_json = df_statements.to_json()

    url = os.getenv('cloudRunURL')
    pred_url = url + '/predict'

    if st.button(label='run spend'):
        response = requests.post(pred_url, json=statement_data_json,
                                 timeout=60)

        st.write(f'response code: {response.status_code} ')
        df_statements['pred_category'] = response.json()['category']
        # flow control side bar
        Side_bar_ran = True
    st.write('Cold start from Cloud run might cause errors, rerun if runs into issue')
    st.write('Dollar Amounts have been changed for privacy')


def data_cleanup(df: pd.DataFrame=None) -> pd.DataFrame:
    '''remove extra fields and mask actual spend'''
    df_stat = df.copy()
    scale_factor = os.getenv('scaleFactor')

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
    df_data = data_cleanup(df=df_statements)
    df_data['Date'] = pd.to_datetime(df_data['Date'])

    current_date = datetime.now()  # Get the current date
    last_month_start = datetime(current_date.year, current_date.month - 1, 1)  # Start of last month
    last_month_end = datetime(current_date.year, current_date.month, 1) - timedelta(days=1)  # End of last month
    two_months_ago_start = datetime(current_date.year, current_date.month - 2, 1)  # Start of two months ago
    two_months_ago_end = datetime(current_date.year, current_date.month - 1, 1) - timedelta(days=1) # End of two months ago
    current_month = current_date.month

    df_last_month = df_data[(df_data['Date'] >= last_month_start) &
                            (df_data['Date'] <= last_month_end)
                            ]

    df_two_months = df_data[(df_data['Date'] >= two_months_ago_start) &
                            (df_data['Date'] <= two_months_ago_end)
                            ]

    df_current = df_data[df_data['Date'].dt.month == current_month]

    # spend 5 by category
    cat1_spend = df_current.groupby("Category")["Amount"].sum()
    cat1_spend = cat1_spend[:6]
    cat1_labels = [f"{x} (${y:.2f})" for x, y in zip(cat1_spend.index, cat1_spend.values)]
    cat1_fig = px.pie(cat1_spend, values=cat1_spend.values, names=cat1_labels, title="Top 5 Category Spend Breakdown")
    cat1_fig.update_traces(textposition='inside', textinfo='label')
    st.plotly_chart(cat1_fig, use_container_width=True)
    st.write(f"Current month so far spend(plus rent): {df_current['Amount'].sum()+2130}")

    # top 5 spend by place
    top_n_by_places = df_current.groupby('Description')['Amount'].sum().nlargest(10)
    top5_fig = px.bar(top_n_by_places, x=top_n_by_places.values,
                      y=top_n_by_places.index, orientation='h',
                      title="Top n Spend by Description")
    top5_fig.update_traces(texttemplate='%{x:.2s}', textposition='outside',
                           text=top_n_by_places.index)
    top5_fig.update_layout(xaxis_title="Spend", yaxis_title="Places",
                           yaxis_tickangle=0, barmode='stack',
                           yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(top5_fig, use_container_width=False)
    
    # current month spend dataframe
    st.dataframe(df_current)
    
    # prediction accuracy

    # last 2 months and current spend comparisons
    months = ['2 months', 'Last', 'Current']
    spends = [df_two_months['Amount'].sum(), df_last_month['Amount'].sum(),
              df_current['Amount'].sum()
              ]
    fig = px.bar({'Spends': spends,
                  'Month': months},
                 y='Month', x='Spends',
                 orientation='h',
                 category_orders={'Month': months},

                 )
    fig.update_traces(texttemplate='%{x:.2s}', textposition='outside',
                      text=months, width=0.4)
    fig.update_layout(xaxis_title="Spend", yaxis_title="Month",
                      yaxis_tickangle=0, barmode='stack')

    st.plotly_chart(fig, use_container_width=False)