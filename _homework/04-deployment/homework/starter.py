#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip freeze | grep scikit-learn')


# change into scikit 1.0.2

get_ipython().system('pip install scikit-learn==1.0.2')


import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# ## Question 1

print(y_pred.mean())


# ## Question 2

year = 2022
month = 2
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df


df_result = df[['ride_id']].copy()
df_result['prediction'] = y_pred
df_result


df_result.to_parquet(
    'fhv_tripdata_2022-02_ride_id_predictions.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


get_ipython().system('ls -lh')


# ## Question 3

get_ipython().system('jupyter nbconvert --to script starter.ipynb')

