import pandas as pd
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
import scipy
import pickle

from fastdownload import download_url

def check_data(data : Path) -> bool:
    print(f"Checking Data {data}")
    return data.exists()

def download_green_taxi_data(year : int, month : int, dest : Path) -> None:
    
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet"
    download_url(url, dest)


def read_dataframe(filename: Path) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


def process_features(df : pd.DataFrame, dv : DictVectorizer = None) -> tuple((scipy.sparse._csr.csr_matrix, scipy.sparse._csr.csr_matrix)):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    target = 'duration'
    
    if dv is None:
        dv = DictVectorizer()
        train_dicts = df[categorical + numerical].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        y_train = df[target].values

        return (dv, X_train, y_train)

    else:

        val_dicts = df[categorical + numerical].to_dict(orient='records')
        X_val = dv.transform(val_dicts)
        y_val = df[target].values

        return (X_val, y_val)
    

def save_pickle(model, output):
    with open(output, 'wb') as f_out:
        pickle.dump(model, f_out)

def load_pickle(file : Path):
    
    with open(file, 'rb') as f_in:
        model = pickle.load(f_in)

    return model