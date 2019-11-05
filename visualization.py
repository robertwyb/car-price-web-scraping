from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
from scipy import stats
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.offline as py
import glob







# eliminate outlier based on z score
# df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]




# webpage part


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Used car price analysis and prediction'),

    html.Div(children=[html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'Ontario', 'value': 'Ontario'},
            {'label': 'Quebec', 'value': 'Quebec'},
            {'label': 'British Columbia', 'value': 'British'},

        ],
        value=['MTL', 'SF'],
        multi=True
    )]),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])


if __name__ == '__main__':
    # app.run_server(debug=True, host='8.8.8.8')
    # create_df()
    df = pd.read_csv('./info.csv', header=0)
    make_type = df['make'].value_counts()
    data = go.Bar(x=make_type.index, y=make_type.values)
    fig = go.Figure(
    data=[go.Bar(y=[2, 1, 3])],
    layout_title_text="A Figure Displayed with fig.show()"
    )
    fig.show()
