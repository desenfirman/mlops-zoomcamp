import pickle
import pendulum
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.logging import get_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(
    name="scheduled-deployment",
    flow_location="./homework.py",
    schedule=CronSchedule(cron="0 9 15 * *"),
)

logger = get_logger()


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(date: str, default_prefixes:str='./data'):
    if date is None:
        date = pendulum.now()
    else:
        date = pendulum.from_format(date, 'YYYY-MM-DD')

    train = date.subtract(months=2)
    valid = date.subtract(months=1)

    train_path = f"{default_prefixes}/fhv_tripdata_{train.format('YYYY-MM')}.parquet"
    valid_path = f"{default_prefixes}/fhv_tripdata_{valid.format('YYYY-MM')}.parquet"
    logger.info(f"Training path is {train_path}")
    logger.info(f"Validation path is {valid_path}")

    df_train = pd.read_parquet(f"{train_path}")
    df_valid = pd.read_parquet(f"{valid_path}")
    logger.info("Data loaded")

    return (df_train, df_valid)
    


@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    categorical = ['PUlocationID', 'DOlocationID']

    df_train, df_val = get_paths(date).result()

    df_train_processed = prepare_features(df_train, categorical)

    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f'models/dv-{date}.b', 'wb') as f_out:
        pickle.dump(dv, f_out)
    
    with open(f'models/model-{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)

    run_model(df_val_processed, categorical, dv, lr)

# main(date="2021-08-15")
