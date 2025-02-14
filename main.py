import pandas as pd
from flask import Flask, request, jsonify
import joblib
import boto3,json,datetime
from io import BytesIO

# Define the CloudWatch log group and stream where logs will be sent
app = Flask(__name__) # __name__ is "__main__" when this script is run directly. It helps Flask to locate resources.
model = None
bucket_name="rentability-forecast"
region = 'us-east-2'

log_group_name = "house-rentability-logs"
log_stream_name = "model-serving-logs"
client = boto3.client('logs',region_name=region)


def load_model():
    '''
    fetches the trained model artifact from the specified S3 bucket and loads it for use
    '''
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key="model-artifact/house_rentability.joblib")
    
    model = joblib.load(BytesIO(response['Body'].read()))
    return model

def log_to_cloudwatch(message):
    '''
    sends logs to AWS CloudWatch
    '''
    timestamp = int((datetime.datetime.now() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)  # current time in milliseconds
    try:
        response = client.put_log_events(
            logGroupName=log_group_name,
            logStreamName=log_stream_name,
            logEvents=[
                {
                    'timestamp': timestamp,
                    'message': json.dumps(message)
                },
            ]
        )
    except client.exceptions.ResourceNotFoundException:
        '''
        error handling: in case the log group or stream does not exist, in which case they are created
        '''
        # If the log group or log stream does not exist, create them
        client.create_log_group(logGroupName=log_group_name)
        client.create_log_stream(logGroupName=log_group_name, logStreamName=log_stream_name)
        log_to_cloudwatch(message)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Loads the trained model.

    Accepts input in JSON format.

    Makes predictions using the model.

    Logs the input data and predictions to CloudWatch.

    Returns the predictions.

    Handles exceptions and logs errors to CloudWatch
    '''
    model = load_model()
    try : 
        input_json = request.get_json()
        df_input = pd.DataFrame([input_json])
        y_predictions = model.predict(df_input)
        response = {'predictions': y_predictions.tolist()}

        log_data = {
            'input': input_json,
            'predictions': response
        }
        log_to_cloudwatch(log_data)

        return jsonify(response), 200
    
    except Exception as e:
        log_to_cloudwatch({'error': str(e)})
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    '''
    starts the Flask application, making it listen on port 5051 and accessible from any IP
    '''
    app.run(debug=True, host="0.0.0.0", port=int(5051))