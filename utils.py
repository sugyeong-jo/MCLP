# Pydeck 사용을 위한 함수 정의
# Shapely 형태의 데이터를 받아 내부 좌표들을 List안에 반환합니다.

import shapely


def line_string_to_coordinates(line_string):
    if isinstance(line_string, shapely.geometry.linestring.LineString):
        lon, lat = line_string.xy
        return [[x, y] for x, y in zip(lon, lat)]
    elif isinstance(line_string, shapely.geometry.multilinestring.MultiLineString): 
        ret = [] 
        for i in range(len(line_string)):
            lon, lat = line_string[i].xy
            for x, y in zip(lon, lat):
                ret.append([x, y])
        return ret


def multipolygon_to_coordinates(x):
    lon, lat = x[0].exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]


def polygon_to_coordinates(x):
    lon, lat = x.exterior.xy
    return [[x, y] for x, y in zip(lon, lat)]


