import pandas as pd
import os
from datetime import datetime

os.environ["AWS_ACCESS_KEY_ID"]="foobar"
os.environ["AWS_SECRET_ACCESS_KEY"]="foobar"
os.environ["S3_ENDPOINT_URL"]="http://127.0.0.1:4566"
os.environ["INPUT_FILE_PATTERN"]="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
os.environ["OUTPUT_FILE_PATTERN"]="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def test_main():
    import batch
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    year = 2021
    month = 1
    input_file = os.getenv('INPUT_FILE_PATTERN').format(year=year, month=month)
    output_file = os.getenv('OUTPUT_FILE_PATTERN').format(year=year, month=month)

    options = {
        'client_kwargs': {
            'endpoint_url': os.getenv('S3_ENDPOINT_URL', None)
        }
    }
    print(options, input_file, output_file)
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    batch.main(year, month, input_file, output_file)