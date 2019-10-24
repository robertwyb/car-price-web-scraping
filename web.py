from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
from scipy import stats
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import folium
from folium.plugins import MarkerCluster
import plotly.graph_objs as go
import plotly.offline as py
import plotly.io as pio
import glob
from sklearn import datasets, linear_model


def create_url(brand, model, number, start_num, used='Used'):
    brand, model = brand.replace(' ', '%20'), model.replace(' ', '%20')
    if model != 'None':
        return f'https://www.autotrader.ca/cars/?rcp={number}&rcs={start_num}&hprc=True&wcp=True&sts' \
           f'={used}&inMarket=basicSearch&mdl={model}&make={brand}'
    return f'https://www.autotrader.ca/cars/?rcp={number}&rcs={start_num}&hprc=True&wcp=True&sts' \
           f'={used}&inMarket=basicSearch&make={brand}'


def selenium_scroll(url):
    driver = webdriver.Chrome('C:/Users/rober/OneDrive/csc/car-price-web-scraping/chromedriver.exe')
    driver.get(url)
    try:
        click = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'btnGotIt'))).click()
        # driver.find_element_by_id('btnGotIt').click()
        driver.refresh()
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    except TimeoutException:
        print("Loading took too much time!")
    return driver.page_source


def scrap_data(brand, model):
    geolocator = Nominatim(user_agent="car_scrapping")
    provinces = ['Ontario', 'Nova Scotia', 'New Brunswick', 'Manitoba', 'British Columbia', 'Québec',
                 'Prince Edward Island', 'Saskatchewan', 'Alberta', 'Northwest Territories',
                 'Newfoundland and Labrador', 'Yukon', 'Nunavut']
    location_info = {}
    data = []
    url = create_url(brand, model, 15, 0)
    print(url)
    result = requests.get(url, headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                                     '(KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'})
    print(result.status_code)
    soup = BeautifulSoup(result.content, 'lxml')
    counts = int(soup.find('span', id='sbCount').text.replace(',', ''))
    print(counts)
    # get all cars
    start_num = 0
    while start_num < counts:
        print('scrapping')
        page_url = create_url(brand, model, 500, start_num)
        start_num += 500
        page_source = selenium_scroll(page_url)
        soup = BeautifulSoup(page_source, 'lxml')
        all_tags = soup.find_all('div', 'col-xs-12 result-item')
        print(len(all_tags))
        for tag in all_tags:
            try:
                full_name = tag.contents[3].contents[3].findChildren('span')[0].text.strip()
                year = full_name[:4]
                # strip out the spec to get model name
                car_name = full_name[5:min([i for i in
                                            [full_name.find(','), full_name.find('|'), full_name.find('('),
                                             full_name.find('+'), full_name.find('*'), full_name.find('/'),
                                             len(full_name)]
                                            if i > -1])]
                car_model = ' '.join(car_name.split()[0:2])
            except:
                car_name = 'Not Available'
                year = None
            try:
                price = tag.contents[3].contents[5].findChildren('span')[0].text.strip()
                price = int(price[1:].replace(',', ''))
            except:
                price = None
            try:
                mileage = tag.contents[3].contents[3].find('div', class_='kms').text.split()[1].replace(',', '')
            except:
                mileage = None
            try:
                city = tag.contents[3].contents[9].findChildren('span')[0].text.strip()
                city = city[:city.find(',')].split()[1:]
                city = ' '.join(city)

            except:
                city = None
            try:
                if city not in location_info:
                    location = geolocator.geocode(city + ' Canada')
                    province = ' '.join([i.strip() for i in location.address.split(',') if i.strip() in provinces])
                    lat, long = location.latitude, location.longitude
                    location_info[city] = {'province': province, 'lat': lat, 'long': long}
                else:
                    province = location_info[city]['province']
                    lat, long = location_info[city]['lat'], location_info[city]['long']
            except:
                province = None
                lat = None
                long = None
            data.append({'year': year, 'make': brand, 'model': car_model, 'sepc': car_name, 'mileage': mileage,
                         'price': price, 'city': city, 'province': province, 'lat': lat, 'long': long})
        # print(data)
    df = pd.DataFrame(data).sort_values(by='year').dropna()
    if model == 'None':
        csv_name = f'./{brand} .csv'
    else:
        csv_name = f'./{brand} {model} .csv'
    df.to_csv(csv_name, index=False, mode='a+')
    print(df)


# concat all csv file into one
def create_df():
    csv_files = glob.glob('*.csv')
    # print(csv_files)
    lst = []
    for filename in csv_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        lst.append(df)
        print('one load')
    frame = pd.concat(lst, axis=0, ignore_index=True)
    frame.to_csv('info.csv', index=False)


# filter dataframe
df = pd.read_csv('info.csv', header=0)
df = df[df['year'] > 2000]
df = df[df['price'] < 500000]
df = df[df['mileage'] < 800000]
df = df.replace('Québec Québec', 'Québec')
dff = df.groupby(['year']).agg({'price': 'mean'})
car_make = df['make'].unique()
df_bar = df.groupby(['make', 'province'], as_index=False).count()
df_bar = df_bar[['make', 'province', 'model']]
bar_data = []
for make in car_make:
    bar_data.append(go.Bar(name=make, x=df_bar.province.unique(),
                         y=df_bar[df_bar['make'] == make]['model'].to_list()))




# ----------------  Create map, only need to run once ----------------------
# trt_coord = (43.664277, -79.391651)
# m = folium.Map(location=trt_coord, zoom_start=12)
# marker_cluster = MarkerCluster().add_to(m)
# for each in df.iterrows():
#     folium.Marker(
#         location = [each[1]['lat'], each[1]['long']],
#         popup=each[1]['model'] + ' ' + each[1]['model'] + ' $' + str(each[1]['price']),
#         icon=folium.Icon(color='green', icon='ok-sign'),
#     ).add_to(marker_cluster)
# m.save('canada_car_map.html')
# ---------------------------------------------------------------------------



# -- web part
external_stylesheets = ['https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


colors = {
    'background': '#F0FFFF',
    'text': '#000000'
    # 'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='Used car price analysis and prediction',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
    ),
    html.Iframe(id='map', srcDoc=open('canada_car_map.html', 'r').read(), width='80%', height='500',
                style={'display': 'block', 'margin-left': 'auto','margin-right': 'auto',
                       'padding':'25px 50px', 'border-style':'none'}),

    html.Div([
        html.Label('Multi-Select Dropdown', style={'color': colors['text'], 'margin': 'auto'}),
        dcc.Dropdown(
            id= 'graph-dropdown',
            options=[
                {'label': 'Ontario', 'value': 'Ontario'},
                {'label': 'Nova Scotia', 'value': 'Nova Scotia'},
                {'label': 'New Brunswick', 'value': 'New Brunswick'},
                {'label': 'Manitoba', 'value': 'Manitoba'},
                {'label': 'British Columbia', 'value': 'British Columbia'},
                {'label': 'Québec', 'value': 'Québec'},
                {'label': 'Prince Edward Island', 'value': 'Prince Edward Island'},
                {'label': 'Saskatchewan', 'value': 'Saskatchewan'},
                {'label': 'Alberta', 'value': 'Alberta'},
                {'label': 'Northwest Territories', 'value': 'Northwest Territories'},
                {'label': 'Newfoundland and Labrador', 'value': 'Newfoundland and Labrador'},
                {'label': 'Yukon', 'value': 'Yukon'},
                {'label': 'Nunavut', 'value': 'Nunavut'}
            ],
            value=['Ontario', 'Nova Scotia', 'New Brunswick', 'Manitoba', 'British Columbia', 'Québec',
                     'Prince Edward Island', 'Saskatchewan', 'Alberta', 'Northwest Territories',
                     'Newfoundland and Labrador', 'Yukon', 'Nunavut'],
            multi=True,
            style={'backgroundColor': colors['background'], 'margin': 'auto' }
        )
        ], style={'backgroundColor': colors['background'], 'margin': 'auto', 'width': '80%'}
    ),

    html.Div(
        html.Div([
            html.Div([
                dcc.Graph(id='mean-price-graph-year', figure={})
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='mean-price-graph-mile', figure={})
            ], className='six columns')
        ], className='row'),
        style={'padding': '50px, 100px', 'width': '80%', 'margin': 'auto'}
    ),


    dcc.Graph(
        id='example-graph',
        figure=go.Figure({
            'data': bar_data,
            'layout': {
                'title': 'Market share based on province',
                'showlegend': True,
                'barmode': 'stack'
            }
        }),
        style={'width':'80%', 'margin': 'auto', 'height': '900px'},
    ),
    html.Div([
    html.Label('Car price prediction', style={'color': colors['text'], 'margin': 'auto'}),
    ])
])


@app.callback(
    Output('mean-price-graph-year', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_year_graph(prov):
    print(prov)
    fig = create_graph(prov, 'year')
    return fig


@app.callback(
    Output('mean-price-graph-mile', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_mile_graph(prov):
    fig = create_graph(prov, 'mileage')
    return fig


def create_graph(prov, ways):
    dff = df.loc[df['province'].isin(prov)]
    dff = dff.groupby([ways]).agg({'price': 'mean'})
# prepare figure_a from value
    figure = {
        'data': [
            {'x': dff.index.values, 'y': dff.price, 'type': 'scatter'},
        ],
        'layout': {
            'title': f'Price based on {ways}'
        }
    }
    print(figure['data'])
    return figure


def lr(model, year, mileage):
    regr = linear_model.LinearRegression()
    df_lr = df[df['model'] == model]
    regr.fit(df_lr[['year', 'mileage']], df_lr['price'])
    df_test = pd.DataFrame([[year, mileage]])
    prediction = regr.predict(df_test[[0, 1]])
    return prediction


if __name__ == '__main__':
    make_list = ['Acura', 'Audi','BMW', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'Ford',
                 'Honda', 'Hyundai', 'Infiniti', 'Jeep', 'Kia', 'Lexus', 'Lincoln',
                 'Mercedes-Benz', 'MINI', 'Nissan', 'Porsche', 'Toyota', 'Volkswagen', 'Volvo']
    model_list = [['bmw', '1 series'], ['bmw', '2 series'], ['bmw', '3 series'], ['bmw', '4 series'],
                 ['bmw', '5 series'], ['bmw', '6 series'], ['bmw', '7 series'], ['bmw', '8 series'],
                 ['bmw', 'x1'], ['bmw', 'x2'], ['bmw', 'x3'], ['bmw', 'x3 M'], ['bmw', 'x4'], ['bmw', 'x4 M'],
                 ['bmw', 'x5'], ['bmw', 'x5 M'], ['bmw', 'x6'], ['bmw', 'x6 M'], ['bmw', 'M'],
                 ['Mercedes-Benz', 'A-Class'], ['Mercedes-Benz', 'B-Class'], ['Mercedes-Benz', 'C-Class'],
                 ['Mercedes-Benz', 'CLA-Class'], ['Mercedes-Benz', 'E-Class'], ['Mercedes-Benz', 'S-Class'],
                 ['Mercedes-Benz', 'GLC-Class'], ['Mercedes-Benz', 'GLE-Class'],
                 ['audi', 'a3'], ['audi', 's3'], ['audi', 'a4'], ['audi', 's4'], ['audi', 'a5'], ['audi', 's5'],
                 ['audi', 'a6'], ['audi', 's6'], ['audi', 'a7'], ['audi', 's7'], ['audi', 'a8'], ['audi', 's8'],
                 ['audi', 'q3'], ['audi', 'q5'], ['audi', 'q7'], ['audi', 'q8']
                 ]

    # for make in make_list:
    #     print(make)
    #     scrap_data(make, 'None')
    #
    # create_df()
    # df = pd.read_csv('info.csv', header=0)
    # make_types = df['make'].value_counts()
    # data = go.Bar(x=make_types.index, y=make_types.values)
    # py.plot([data])
    app.run_server(debug=True)


