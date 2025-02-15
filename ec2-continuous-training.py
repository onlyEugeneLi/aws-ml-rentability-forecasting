import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import boto3,logging,sys,time
from io import BytesIO

region = "us-east-2"
bucket_name = "rentability-forecast"
file_name = "house_rental_evaluation.csv"
input_data_file_path = f"s3://{bucket_name}/{file_name}"

LOG_GROUP = 'house-rentability-model-retraining'
LOG_STREAM = 'model-training-stream'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

client = boto3.client('logs',region_name=region)

def ensure_log_stream_exists():
    try:
        client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=LOG_STREAM)
    except client.exceptions.ResourceAlreadyExistsException:
        pass

def log_message_to_cloudwatch(message):

    streams = client.describe_log_streams(logGroupName=LOG_GROUP, logStreamNamePrefix=LOG_STREAM)
    if not streams['logStreams']:
        # If not, create it
        client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=LOG_STREAM)
        sequence_token = None
    else:
        # If it exists, retrieve the sequenceToken
        sequence_token = streams['logStreams'][0].get('uploadSequenceToken')
    
    # Create the log event
    log_event = {
        'timestamp': int(round(time.time() * 1000)),
        'message': message
    }

    if sequence_token:
        response = client.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=[log_event],
            sequenceToken=sequence_token
        )
    else:
        response = client.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=[log_event]
        )

    return response

def load_data(file_path):
    df_re = pd.read_csv(file_path)
    return df_re

def validate_data(df):
    EXPECTED_COLUMNS = [
    "ADS_ID", "ADS_CATEGORY_NAME", "ADS_FIRST_PUBLICATION_DATE", "ADS_SUBJECT",
    "ADS_PRICE", "ADS_OPT_URGENT", "ADS_OWNER_TYPE", "ADS_ATTR_REAL_ESTATE_TYPE",
    "ADS_ATTR_ROOMS", "ADS_ATTR_SQUARE", "ADS_ATTR_GES", "ADS_ATTR_ENERGY_RATE",
    "ADS_ATTR_FURNISHED", "ADS_GEO_LAT", "ADS_GEO_LNG", "ADS_GEO_CITY", "ADS_GEO_ZIPCODE",
    "ADS_GEO_REGION", "ADS_GEO_DEPARTEMENT", "ADS_GEO_ARRONDISSEMENT",
    "ADS_GEO_ARRONDISSEMENT_LAT", "ADS_GEO_ARRONDISSEMENT_LNG"
    ]
    if set(df.columns) != set(EXPECTED_COLUMNS):
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
        
        if missing_cols:
            logger.error(f"Missing columns in input data: {missing_cols}")
        if extra_cols:
            logger.error(f"Unexpected columns in input data: {extra_cols}")
        
        return False
    return True

def preprocess_data(df):
    df = df[df["ADS_CATEGORY_NAME"] == "Locations"]
    df["ADS_PRICE_SQUARE"] = df["ADS_PRICE"] / df["ADS_ATTR_SQUARE"]

    df_grouped = df.groupby(["ADS_GEO_ZIPCODE", "ADS_GEO_CITY", "ADS_ATTR_REAL_ESTATE_TYPE"])
    df_aggregated = df_grouped.agg(COUNT_ADS=("ADS_ID", "count"), MED_PRICE=("ADS_PRICE_SQUARE", "median"))
    df = df.merge(df_aggregated, on=["ADS_GEO_ZIPCODE", "ADS_GEO_CITY", "ADS_ATTR_REAL_ESTATE_TYPE"], how="inner")

    df = df[
        (df["ADS_PRICE_SQUARE"] / (df["ADS_PRICE_SQUARE"] + df["MED_PRICE"]) >= 0.25) &
        (df["ADS_PRICE_SQUARE"] / (df["ADS_PRICE_SQUARE"] + df["MED_PRICE"]) < 0.75) &
        (df["ADS_PRICE_SQUARE"] < 150) &
        (df["ADS_PRICE_SQUARE"] > 0) &
        (df["ADS_ATTR_SQUARE"] >= 9) &
        (df["ADS_ATTR_SQUARE"] <= 300) &
        (df["COUNT_ADS"] >= 5)
    ]
    return df

def encode_attributes(df):
    def ADS_ATTR_FURNISHED_Encode_Python(x):
        if x == "Meublé":
            return 2
        elif x == "Non meublé":
            return 1
        else:
            return 0

    def ADS_ATTR_REAL_ESTATE_TYPE_Encode_Python(x):
        if x == "Maison":
            return 2
        elif x == "Appartement":
            return 1
        else:
            return 0

    df["ADS_ATTR_FURNISHED_NUM"] = df["ADS_ATTR_FURNISHED"].apply(ADS_ATTR_FURNISHED_Encode_Python)
    df["ADS_ATTR_REAL_ESTATE_TYPE_NUM"] = df["ADS_ATTR_REAL_ESTATE_TYPE"].apply(ADS_ATTR_REAL_ESTATE_TYPE_Encode_Python)

    # Select the final columns
    df_final = df[["ADS_GEO_LAT", "ADS_GEO_LNG", "ADS_ATTR_ROOMS", "ADS_ATTR_SQUARE",
                   "ADS_ATTR_REAL_ESTATE_TYPE_NUM", "ADS_ATTR_FURNISHED_NUM", "ADS_PRICE"]]

    return df_final

def train_model(X_train, y_train,n_estimators=250):

    model = XGBRegressor(
        booster='gbtree',
        objective='reg:squarederror',
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        gamma=1,
        subsample=0.75,
        colsample_bytree=0.75,
        scale_pos_weight=1,
        n_jobs=-1,
        verbosity=1,
        n_estimators=n_estimators
    )

    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=True)
    return model

def predict(model, X_test):
    pred_test = model.predict(X_test)
    return pred_test

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return r2_score(y_test, predictions)

def save_model_to_s3(model, bucket_name,file_key):
    buffer = BytesIO()
    joblib.dump(model, buffer)
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=buffer.getvalue())

df = load_data(input_data_file_path)

if not validate_data(df):
    logger.error("Data validation failed. Exiting the script.")
    sys.exit(1)

df_preprocessed = preprocess_data(df)
df_final = encode_attributes(df_preprocessed)

target = "ADS_PRICE"
predictors = [x for x in df_final.columns if x not in [target]]

X_train, X_test, y_train, y_test = train_test_split(df_final[predictors], df_final[target], test_size=0.1)
xgb_model = train_model(X_train, y_train)

R2_THRESHOLD = 0.7

r2 = evaluate_model(xgb_model, X_test, y_test)

if r2 >= R2_THRESHOLD:
    success_msg = "Model meets the performance threshold. Saving to S3..."
    print(success_msg)
    log_message_to_cloudwatch(success_msg)
    model_file_path = "house_rentability.joblib"
    save_model_to_s3(xgb_model, bucket_name, "model-artifact/house_rentability.joblib")
else:
    failure_msg = f"Model does not meet the performance threshold. Not saving. R2 Score: {r2}"
    print(failure_msg)
    log_message_to_cloudwatch(failure_msg)