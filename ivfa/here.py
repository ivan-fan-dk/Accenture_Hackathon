import polars as pl
import numpy as np
from scipy.spatial.distance import cdist
df = pl.read_parquet(r"../dataset/data.parquet")
df.head()
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
    """
    Filters the DataFrame and ensures the resulting feature matrix rows
    are in the *exact same order* as the input_track_ids list.
    """
    # 1. Create a DataFrame from the input IDs to establish the desired order
    df_order = pl.DataFrame({"track_id": input_track_ids}).with_row_index("order_index")

    # 2. Join the main DataFrame to the order DataFrame
    #    An inner join ensures only tracks in input_track_ids are kept.
    df_ordered_tracks = df_order.join(df, on="track_id", how="left")

    # 3. Sort by the temporary index column to restore the original list order
    df_ordered_tracks = df_ordered_tracks.sort("order_index")

    # 4. Select features and convert to NumPy array
    return get_feature_vector(df_ordered_tracks, feature_columns)
def get_input_track_ids_df(df: pl.DataFrame, input_track_ids: list[str]) -> list[str]:
    return df.filter(pl.col('track_id').is_in(input_track_ids))

artists = input_track_ids_df.select('artists').to_numpy().flatten().tolist()
artists_flat = [a.strip() for s in artists for a in s.split(';') if a.strip()]
artists_flat