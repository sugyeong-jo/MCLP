#%%
import os
from typing import Dict

import pandas as pd
import geopandas as gpd
import numpy as np
import yaml
# import mclp
from shapely.geometry import Polygon, Point
import pydeck as pdk

from utils import *


def load_config(config_filename: str, **kwargs) -> Dict:
    with open(config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


# Set the viewport location
center = [127.696280, 34.940640]
view_state = pdk.ViewState(
    longitude=center[0],
    latitude=center[1],
    zoom=10
)


#%%
selected_number = [1, 3, 6, 8, 10, 11, 12, 13, 14, 20] # 11, 12, 13,

config_file = load_config('configure.yaml')
for currentdir, dirs, files in os.walk(config_file['DATA_PATH']):
    for index, file in enumerate(files):
        if index+1 in selected_number:
            if 'csv' in file:
                variable = f'df_{str(index+1).zfill(2)}'
                print(variable, file)
                globals()[variable] = pd.read_csv(currentdir+file)
            if 'geojson' in file:
                variable = f'df_{str(index+1).zfill(2)}'
                print(variable, file)
                globals()[variable] = gpd.read_file(currentdir+file)                

#%%
# 1. 격자별 인구 현황

# 쉬운 분석을 위한 임의의 grid id 부여
df_08['grid_id'] = 0
idx = []
for i in range(len(df_08)):
    idx.append(str(i).zfill(5))
df_08['grid_id'] = pd.DataFrame(idx)

df_08['val'] = df_08['val'].fillna(0)
df_08['정규화인구'] = df_08['val'] / df_08['val'].max()

# pydeck 을 위한 coordinate type, geometry를 coordinate 형태로 적용
df_08['coordinates'] = df_08['geometry'].apply(multipolygon_to_coordinates)

df_08['coord_cent'] = 0
df_08['geo_cent'] = 0
temp_0 = []
temp_1 = []
for i in df_08['geometry']:
    cent = [[i[0].centroid.coords[0][0], i[0].centroid.coords[0][1]]]
    temp_0.append(cent)
    temp_1.append(Point(cent[0]))
df_08['coord_cent'] = pd.DataFrame(temp_0)  # pydeck을 위한 coordinate type
df_08['geo_cent'] = temp_1  # geopandas를 위한 geometry type

# 인구 현황이 가장 높은 위치
df_08.iloc[df_08["val"].sort_values(ascending=False).index].reindex().head()

#%%
# 2. 자동차 등록대수
# 격자별 자동차 등록대수
df_03['totale'].fillna(0)
df_03['coordinates'] = df_03['geometry'].apply(polygon_to_coordinates) #pydeck 을 위한 coordinate type
df_03.iloc[df_03["totale"].sort_values(ascending=False).index].reindex().head()
#%%
# 3. 전기자동차
# 전기차 등록 대수 점 수 부여
# 년도 별, 행정구역 별, 전기차 보급 추세
list_EV_dist = pd.merge(
    pd.merge(
        df_06[df_06["기준년도"] == 2017][['행정구역', '보급현황']],
        df_06[df_06["기준년도"] == 2018][['행정구역', '보급현황']],
        how='outer', on='행정구역'
        ), pd.merge(
            df_06[df_06["기준년도"] == 2019][['행정구역', '보급현황']],
            df_06[df_06["기준년도"] == 2020][['행정구역', '보급현황']],
            how='outer', on='행정구역'
            ), how='outer', on='행정구역'
        )

list_EV_dist.columns = ["ADM_DR_NM", "2017", "2018", "2019", "2020"]
list_EV_dist = list_EV_dist.iloc[
    list_EV_dist[["ADM_DR_NM", "2017", "2019", "2020"]].mean(axis=1).sort_values(ascending=False).index
    ].reindex()
# 2020년 기준으로 가장 많은 비율을 차지하는 광양읍에 전체적으로 점수를 크게 부여할 것
df_EV_ADM = pd.merge(list_EV_dist, df_20, on="ADM_DR_NM")
# list_EV_dist[["행정구역", "2017","2019","2020"]].mean(axis=1)
df_EV_ADM

#%%
df_11_time7
#%%
#4. 교통량
df = df_10
df['coordinate'] = df['geometry'].buffer(0.001).apply(polygon_to_coordinates) 
df = pd.DataFrame(df) # geopanadas 가 아닌 pandas 의 데이터프레임으로 꼭 바꿔줘야 합니다. 
df['정규화도로폭'] = df['width'].apply(int) / df['width'].apply(int).max()
# 대부분의 사람은 7시에 주거지역에서 업무지역으로 움직일 것으로 가정
# 승용차만 고려
df_11_time7 = df_11[df_11['시간적범위']==7]

df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time7[df_11_time7['link_id'].apply(str).str.contains(i)]['승용차'])])    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time7=pd.merge(df, df_10_,on='link_id' )
#%%
# 교통량 합이 가장 높은 도로
df_10_11_time7.iloc[df_10_11_time7["교통량"].sort_values(ascending=False).index].reindex().head()

# 대부분의 사람은 오후 3시에 업무를 하는 것으로 가정 (운송 업 포함)
df_11_time15=df_11[df_11['시간적범위']==15]
df_10_ = []
for i in df_10.link_id:
    df_10_.append([i,sum(df_11_time15[df_11_time15['link_id'].apply(str).str.contains(i)]['승용차'])])    
df_10_ = pd.DataFrame(df_10_).fillna(0)
df_10_.columns = ["link_id", "교통량"]
df_10_11_time15=pd.merge(df, df_10_,on = 'link_id' )

# 교통량 합이 가장 높은 도로
df_10_11_time15.iloc[df_10_11_time15["교통량"].sort_values(ascending=False).index].reindex().head()
#%%
# Make layer
# 사람이 있는 그리드만 추출
layer = pdk.Layer(
    'PolygonLayer',  # 사용할 Layer 타입
    df_08[(df_08['val'].isnull() == False) & df_08['val'] != 0],  # 시각화에 쓰일 데이터프레임
    get_polygon='coordinates',  # geometry 정보를 담고있는 컬럼 이름
    get_fill_color='[0, 255*정규화인구, 0, 정규화인구*10000 ]',  # 각 데이터 별 rgb 또는 rgba 값 (0~255)
    pickable=True,  # 지도와 interactive 한 동작 on
    auto_highlight=True  # 마우스 오버(hover) 시 박스 출력
    )
# Render
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    # map_style='mapbox://styles/mapbox/streets-v11',
    # mapbox_key="pk.eyJ1IjoiamNsYXJhODExIiwiYSI6ImNrdHViMnU5bDF5ODUyeXBuNmUxNXZ2YXgifQ.5ZnpOFIuIQuvDx5atrHgRw"
)

#r.to_html()
r.show()
#%%
# 격자별 자동차 등록대수
# val 열 na 제거
df_03['totale'].fillna(0)
# coordinate 
df_03['coordinates'] = df_03['geometry'].apply(polygon_to_coordinates) #pydeck 을 위한 coordinate type
# 인구 현황이 가장 높은 위치
df_03.iloc[df_03["totale"].sort_values(ascending=False).index].reindex().head()

#%%
# 전기차 등록 대수 점 수 부여
#년도 별, 행정구역 별, 전기차 보급 추세
list_EV_dist = pd.merge(pd.merge(df_06[df_06["기준년도"]==2017][['행정구역', '보급현황']],                                 
                                 df_06[df_06["기준년도"]==2018][['행정구역', '보급현황']],
                                 how = 'outer', on = '행정구역'),

                                 pd.merge(df_06[df_06["기준년도"]==2019][['행정구역', '보급현황']],
                                 df_06[df_06["기준년도"]==2020][['행정구역', '보급현황']],
                                 how = 'outer', on = '행정구역'),
                                 how = 'outer', on = '행정구역'
                                )

list_EV_dist.columns  = ["ADM_DR_NM", "2017", "2018","2019","2020"]
list_EV_dist=list_EV_dist.iloc[list_EV_dist[["ADM_DR_NM", "2017","2019","2020"]].mean(axis=1).sort_values(ascending=False).index].reindex()
# 2020년 기준으로 가장 많은 비율을 차지하는 광양읍에 전체적으로 점수를 크게 부여할 것
df_EV_ADM = pd.merge(list_EV_dist, df_20, on = "ADM_DR_NM")

#list_EV_dist[["행정구역", "2017","2019","2020"]].mean(axis=1)
df_EV_ADM

#%%
    radius = radius = (1/88.74/1000)*i
    K = 40
    M = 700
    opt_sites_org,f = mclp(np.array(points),K,radius,M,df_result_fin,w,'w_미세먼지')
    미세먼지iot_40개.append([i,f/len(points)])