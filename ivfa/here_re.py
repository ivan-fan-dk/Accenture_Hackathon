import polars as pl
import numpy as np
from scipy.spatial.distance import cdist
import json  # 导入 json 库以便更美观地打印字典

# --- 1. 模拟数据 ---
# 我们不再读取 "data.parquet"，而是直接使用你提供的 CSV 示例
# 这样代码就可以在任何地方运行
from io import StringIO
csv_data = """
,track_id,artists,album_name,track_name,popularity,duration_ms,explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,time_signature,track_genre
0,5SuOikwiRyPMVoIQDJUgSV,Gen Hoshino,Comedy,Comedy,73,230666,False,0.676,0.461,1,-6.746,0,0.143,0.0322,1.01e-06,0.358,0.715,87.917,4,acoustic
1,4qPNDBW1i3p13qLCt0Ki3A,Ben Woodward,Ghost (Acoustic),Ghost - Acoustic,55,149610,False,0.42,0.166,1,-17.235,1,0.0763,0.924,5.56e-06,0.101,0.267,77.489,4,acoustic
2,1iJBSr7s7jYXzM8EGcbK5b,Ingrid Michaelson;ZAYN,To Begin Again,To Begin Again,57,210826,False,0.438,0.359,0,-9.734,1,0.0557,0.21,0.0,0.117,0.12,76.332,4,acoustic
3,6lfxq3CG4xtTiEg7opyCyx,Kina Grannis,Crazy Rich Asians (Original Motion Picture Soundtrack),Can't Help Falling In Love,71,201933,False,0.266,0.0596,0,-18.515,1,0.0363,0.905,7.07e-05,0.132,0.143,181.74,3,acoustic
4,5vjLSffimiIP26QG5WcN2K,Chord Overstreet,Hold On,Hold On,82,198853,False,0.618,0.443,2,-9.681,1,0.0526,0.469,0.0,0.0829,0.167,119.949,4,acoustic
5,01MVOl9KtVTNfFiBU9I7dc,Tyrone Wells,Days I Will Remember,Days I Will Remember,58,214240,False,0.688,0.481,6,-8.807,1,0.105,0.289,0.0,0.189,0.666,98.017,4,acoustic
6,6Vc5wAMmXdKIAM7WUoEb7N,A Great Big World;Christina Aguilera,Is There Anybody Out There?,Say Something,74,229400,False,0.407,0.147,2,-8.822,1,0.0355,0.857,2.89e-06,0.0913,0.0765,141.284,3,acoustic
7,1EzrEOXmMH3G43AXT1y7pA,Jason Mraz,We Sing. We Dance. We Steal Things.,I'm Yours,80,242946,False,0.703,0.444,11,-9.331,1,0.0417,0.559,0.0,0.0973,0.712,150.96,4,acoustic
8,0IktbUcnAGrvD03AWnz3Q8,Jason Mraz;Colbie Caillat,We Sing. We Dance. We Steal Things.,Lucky,74,189613,False,0.625,0.414,0,-8.7,1,0.0369,0.294,0.0,0.151,0.669,130.088,4,acoustic
9,7k9GuJYLp2AzqokyEdwEw2,Ross Copperman,Hunger,Hunger,56,205594,False,0.442,0.632,1,-6.77,1,0.0295,0.426,0.00419,0.0735,0.196,78.899,4,acoustic
10,4mzP5mHkRvGxdhdGdAH7EJ,Zack Tabudlo,Episode,Give Me Your Forever,74,244800,False,0.627,0.363,8,-8.127,1,0.0291,0.279,0.0,0.0928,0.301,99.905,4,acoustic
"""
# 模拟的 DataFrame
df = pl.read_csv(StringIO(csv_data), ignore_errors=True).drop("")


# --- 2. 你的原始变量和函数 (无修改) ---
FEATURE_COLUMNS = [
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
]

input_track_ids = ["5SuOikwiRyPMVoIQDJUgSV",
        "4qPNDBW1i3p13qLCt0Ki3A",
        "1iJBSr7s7jYXzM8EGcbK5b",
        "6lfxq3CG4xtTiEg7opyCyx",
        "5vjLSffimiIP26QG5WcN2K",
        "01MVOl9KtVTNfFiBU9I7dc",
        "6Vc5wAMmXdKIAM7WUoEb7N",
        "1EzrEOXmMH3G43AXT1y7pA",
        "0IktbUcnAGrvD03AWnz3Q8",
        "7k9GuJYLp2AzqokyEdwEw2",
        "4mzP5mHkRvGxdhdGdAH7EJ",]

def get_feature_vector(df: pl.DataFrame, feature_columns: list[str]) -> np.array:
    return df.select(feature_columns).to_numpy()
def get_array_input(df: pl.DataFrame, input_track_ids: list[str], feature_columns: list[str]) -> np.array:
    df_input_tracks = df.filter(pl.col('track_id').is_in(input_track_ids))
    return get_feature_vector(df_input_tracks, feature_columns)
def get_array_input_ordered(df: pl.DataFrame, input_track_ids: list[str], feature_columns: list[str]) -> np.array:
    df_order = pl.DataFrame({"track_id": input_track_ids}).with_row_index("order_index")
    df_ordered_tracks = df_order.join(df, on="track_id", how="left")
    df_ordered_tracks = df_ordered_tracks.sort("order_index")
    return get_feature_vector(df_ordered_tracks, feature_columns)
def get_input_track_ids_df(df: pl.DataFrame, input_track_ids: list[str]) -> pl.DataFrame:
    return df.filter(pl.col('track_id').is_in(input_track_ids))

# --- 3. 修正后的 'artists' 提取 ---
print("--- (修正后) 仅 artists 列表 (展平) ---")
df_input_tracks = get_input_track_ids_df(df, input_track_ids)
artists_list = df_input_tracks.select('artists').to_series().to_list()
artists_flat = sorted(list(set([a.strip() for s in artists_list for a in s.split(';') if a.strip()]))) # 排序并去重
print(artists_flat)

print("\n" + "="*30 + "\n")


# --- 4. 你要求的新功能：获取所有非浮点列的唯一值 ---

def get_unique_non_float_values(df: pl.DataFrame, input_track_ids: list[str]) -> dict:
    """
    根据 input_track_ids 筛选 df, 
    然后返回所有非浮点列的唯一值字典。
    """
    
    # 1. 筛选数据
    df_inputs = get_input_track_ids_df(df, input_track_ids)
    
    # 2. 识别非浮点列
    non_float_cols = []
    for col_name, dtype in df_inputs.schema.items():
        if not dtype.is_float():
            non_float_cols.append(col_name)
            
    # 3. 提取唯一值
    unique_results = {}
    for col_name in non_float_cols:
        
        # --- 对 'artists' 列进行特殊处理 (拆分和展平) ---
        if col_name == 'artists':
            all_artists_strings = df_inputs.select('artists').to_series().to_list()
            flat_list = [a.strip() for s in all_artists_strings for a in s.split(';') if a.strip()]
            uniques = sorted(list(set(flat_list))) # 排序并去重
            unique_results[col_name] = uniques
        
        # --- 对所有其他非浮点列的标准处理 ---
        else:
            uniques = (
                df_inputs.select(col_name)
                .unique()      
                .sort(col_name)  
                .to_series()   
                .to_list()     
            )
            # 过滤掉 None/null 值 (如果存在)
            unique_results[col_name] = [item for item in uniques if item is not None]
            
    return unique_results

# --- 5. 执行新功能并打印结果 ---

print("--- 所有非浮点列的唯一值 (已实现你的需求) ---")
unique_values_map = get_unique_non_float_values(df, input_track_ids)
# 使用 json.dumps 可以让字典打印得更整齐
print(json.dumps(unique_values_map, indent=2, ensure_ascii=False))