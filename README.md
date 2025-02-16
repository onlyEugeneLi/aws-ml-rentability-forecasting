# Develop and deploy an XGBoost House Rentability Forecasting model using AWS EC2 and API Gateway

This project is based on the [lab session](https://e2.udemymail.com/ls/click?upn=u001.SYmHo8n5cEkWdfJ1quYpib1GpxNSnHXn-2BTfowh5U-2BJpl9WAzvlNzKsDmEx5ZZKkbhhdwd592K3geZGQGYdhXe-2BwqirRbGxY1L-2FyPUA3wQ8pdKa1Q6T0iBEadaZstbSf7pOtDiizd0mBMjq50XoWWxE0Zgru73dwdo3fgQ6SxoiKez6XGkGaFoM4tmU7yHW0MRuN-2FtUm39OkpPLYjnnXuvqs3nwu6yiA2L7NApuXA5GpX7veszj0HObGv1Ic7b4E4Tnx-2FZQMuubG-2BN0hsAUl61w-3D-3D7IYO_2ufKv4pZrDqayQ700wQvleH4JEzHGS1DtwRZRKW4nCK2nwGwB-2BsHUYwDr2Gn4-2BFTgPHkMBITTKwvBzlZZZI7mP9An9kI0ILrHyXT4Cxe5FAm-2F8eXjeNC7AyAg0XkT-2F-2BnY7mH2VzQMtXMVz-2BCwl6ZdgItfXgkPWUaZ8qefdiTPIqD7O-2Fs3ueJgSuYvL0Z81eVcERFZuRt4WiVbjEzx854YhDYeIMXz6Ups7qWAW9g6t-2FTpL79SMDYDgtu-2FlazIxoC0X3-2F-2F0cbwUBgDzxKpPjxwOT9h4oIQxG7MBfWtWZ0boyGOssIEtLUcQmV669s4JXhVxXdmckfO73-2BHY2PZZuWTvitEq7QnsKKZ97FbNFu6x9LTjSNHxOr9txb5odBlxcP) in Udemy. 

<center><img src='https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fi%2F5p94qxqe2w9vzchif3ll.jpg' alt='' width='400' align='center'>
</center>

## Try it out!

For **MacOS/Linux** users, run this command in your terminal:

```
curl -X POST https://zgkg0grdt0.execute-api.us-east-2.amazonaws.com/production/ml-api-resource \
-H "x-api-key: iwYYU1UI1u2SsTqsLobuH2fxqYQqKUCK1emjmFZX" \
-H "Content-Type: application/json" \
-d '{"ADS_GEO_LAT":50.63334,"ADS_GEO_LNG":3.04214,"ADS_ATTR_ROOMS":1,"ADS_ATTR_SQUARE":20,"ADS_ATTR_REAL_ESTATE_TYPE_NUM":1,"ADS_ATTR_FURNISHED_NUM":2}'
```

Feel free to tweak the feature values to see how the predictions change!

For **Windows** users, 

1. Prepare data in JSON format

    In the Prompt Command's working directory

    Create a new `.json` file and name it `test-data.json`

    Copy and Paste the following data into the file

    ```
    {"ADS_GEO_LAT":50.63334,"ADS_GEO_LNG":3.04214,"ADS_ATTR_ROOMS":1,"ADS_ATTR_SQUARE":20,"ADS_ATTR_REAL_ESTATE_TYPE_NUM":1,"ADS_ATTR_FURNISHED_NUM":2}
    ```

1. Run this commnad in your Powershell or Prompt Command:

    ```
    curl.exe -X POST https://zgkg0grdt0.execute-api.us-east-2.amazonaws.com/production/ml-api-resource -H "x-api-key: iwYYU1UI1u2SsTqsLobuH2fxqYQqKUCK1emjmFZX" -H "Content-Type: application/json" -d @test-data.json
    ```

**Result**: 

You should expect something like this returned back from the API. 

```
{
  "message": "Thank you for trying my first ML model in production.",
  "predictions": [
    495.02734375
  ]
}
```

## Background 

As the housing market continuously evolves due to factors like economic shifts, policy changes, and regional developments, it is benefitial to stay ahead by developing a predictive service that forecasts the rentability of houses. The house rental dataset covers rental market data across different U.S. regions. 

It is commonly agreed that leveraging cloud technology and machine learning can offer a competitive edge and the goal of this project is to leverage AWS cloud and machine learning to forecast house rentability across various U.S. regions, enhancing my data-driven project portfolio.

<center><img src='https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png' alt='xgboost logo' width='200' align='center'></center>

## Objectives

* Perform EDA of House rentability data using Jupyter Notebooks on AWS EC2
* Train & deveop a Data Science Regression Model using XGBoost & Pandas
* Deploy Model Serving Flask Application on AWS EC2
* Perform Model Evaluation and Data Validation for Continuous Model Training
* Use CloudWatch Logs for Model Training Metrics and Logs
* Deploy Continuous Training job on newly arriving batches of data on AWS EC2


## Project Environment Setup

1. **SSH Key setup**

    Modify the file's permissions using the chmod command to enhance security.

    ```
    chmod 400 aws-instance-keypair.pem
    ```

1. **Acccessing the EC2 Instance**

    Use the terminal to SSH into your EC2. This will require your **.pem** file and the EC2's **Public IPv4 DNS**.

    ```
    ssh -i "file_name.pem" ubuntu@public-ipv4-dns.com 
    ```

    Prompted message:

    ```
    The authenticity of host 'ec2-instance-public-IPv4-DNS.com' can't be established.
    DS36001 key fingerprint is SHA256:aaaaaaaaaaasssssssdddddd.
    This key is not known by any other names
    Are you sure you want to continue connecting (yes/no/[fingerprint])?
    ```

    Enter `yes`

1. **Setting Up the EC2 Environment**

    Update the package list on your EC2.

    Install Python tools, such as pip and the virtual environment.

    Install the following specific Python libraries with the mentioned versions:

    Xgboost (version 1.6.2), pandas (version 1.5.2), scikit-learn (version 1.1.1), matplotlib (version 3.5.2), seaborn, boto3, fsspec, s3fs (version 2023.9.2), Flask

    Prepare environment:

    ```
    sudo apt update
    sudo apt upgrade
    sudo apt install python3-pip # python3.12 comes with ubuntu image 
    sudo apt install python3-virtualenv
    sudo apt install pipx
    pipx ensurepath # Added /home/ubuntu/.local/bin to the PATH environment variable
    # /home/ubuntu/.local/bin' is not on your PATH environment variable
    echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
    # shell's config file (i.e. ~/.bashrc)
    source ~/.bashrc
    
    ```

    Create and activate environment:

    ```
    virtualenv venv_name
    source venv_name/bin/activate
    ```

    In virtual environmnt, download dependencies: (Can download the latest version)

    ```
    pip install xgboost==1.6.2 pandas==1.5.2 joblib==1.2.0 scikit-learn==1.1.1 matplotlib==3.5.2 boto3 fsspec s3fs==2023.9.2 seaborn 
    
    sudo apt install python3-flask
    ```

1. **Installing Jupyter**

    ```
    pip install jupyterlab
    
    pip install jupyter notebook
    
    echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
    
    source ~/.bashrc

    # Set password
    jupyter notebook password
    ```

    Check Linux system disk space

    ```
    df -h
    ```

1. **Configuring Security on AWS**

1. **Starting Jupyter Lab on EC2**

    Initiate Jupyter Notebook inside EC2 instance

    ```
    jupyter-lab --ip 0.0.0.0 --no-browser --allow-root
    ```

1. **Accessing Jupyter Lab Externally**

    Go to public IPv4 address:port number/lab

    Log in with Jupyter Lab password

## Model Serving Application using Flask & EC2

Write a Flask script which will be deployed in your **EC2** instance to serve predictions from the serialized machine learning model stored in your S3 bucket . This service, listening on port 5051, offers a /predict endpoint that processes input data, logs activities to CloudWatch, and handles errors efficiently. 

Steps:

1. Set up a **Flask** web application to serve a machine learning model.

1. Load a trained model artifact from an **S3** bucket named udemy-ds-lab.

1. Implement an endpoint /predict that:

    * Accepts input data in JSON format.

    * Predicts outcomes using the loaded model.

    * Logs both input data and predictions to **CloudWatch**.

1. Handle errors gracefully by logging them to **CloudWatch** and returning appropriate error messages.

1. Ensure the application listens on port 5051 and is accessible from any IP.

**Upload the Flask script main.py to the folder flask_app:**

```
scp -i "name-of-keypair.pem" flask-script/main.py ubuntu@your-ec2-public-ipv4-dns:flask_app
```

**Run a curl command to get predictions against the model serving Flask application running on port 5051 of your EC2 instance.**

```
curl -X POST http://your-ec2-instance-public-ip:5051/predict \
-H "Content-Type: application/json" \
-d '{"ADS_GEO_LAT":50.63334,"ADS_GEO_LNG":3.04214,"ADS_ATTR_ROOMS":1,"ADS_ATTR_SQUARE":20,"ADS_ATTR_REAL_ESTATE_TYPE_NUM":1,"ADS_ATTR_FURNISHED_NUM":2}'
```

**Kill the Flask application & run the Flask application in the background this time by executing a `nohup` command**

```
nohup python flask_app/main.py > model_serving_logs.txt 2>&1 &
```

To check all the running processes:

```
ps -ef
```

To kill the process:

```
kill PID
```

## Create API Gateway proxy for model serving endpoint

1. Define a new REST API

1. Resource & method: add a new resource and create a POST method for it. 

1. Validation: Test the POST method with a sample JSON payload to ensure it's working as expected.

    ```
    {"ADS_GEO_LAT":50.63334,"ADS_GEO_LNG":3.04214,"ADS_ATTR_ROOMS":1,"ADS_ATTR_SQUARE":20,"ADS_ATTR_REAL_ESTATE_TYPE_NUM":1,"ADS_ATTR_FURNISHED_NUM":2}
    ```

1. Deployment: Deploy the API to a new stage.

1. Access Details: After deployment, obtain the Invoke URL for external testing purposes.

1. External Test: Use the curl command to test the API with the provided JSON payload.

    ```
    curl -X POST https://zgkg0grdt0.execute-api.us-east-2.amazonaws.com/production/ml-api-resource \
    -H "x-api-key: iwYYU1UI1u2SsTqsLobuH2fxqYQqKUCK1emjmFZX" \
    -H "Content-Type: application/json" \
    -d '{"ADS_GEO_LAT":50.63334,"ADS_GEO_LNG":3.04214,"ADS_ATTR_ROOMS":1,"ADS_ATTR_SQUARE":20,"ADS_ATTR_REAL_ESTATE_TYPE_NUM":1,"ADS_ATTR_FURNISHED_NUM":2}'
    ```

    One-line command (Linux):

    ```
    curl -X POST https://zgkg0grdt0.execute-api.us-east-2.amazonaws.com/production/ml-api-resource -H "x-api-key: iwYYU1UI1u2SsTqsLobuH2fxqYQqKUCK1emjmFZX" -H "Content-Type: application/json" -d '{"ADS_GEO_LAT":50.63334,"ADS_GEO_LNG":3.04214,"ADS_ATTR_ROOMS":1,"ADS_ATTR_SQUARE":20,"ADS_ATTR_REAL_ESTATE_TYPE_NUM":1,"ADS_ATTR_FURNISHED_NUM":2}'
    ```

    One-line command (Windows):

    ```
    curl.exe -X POST https://zgkg0grdt0.execute-api.us-east-2.amazonaws.com/production/ml-api-resource -H "x-api-key: iwYYU1UI1u2SsTqsLobuH2fxqYQqKUCK1emjmFZX" -H "Content-Type: application/json" -d @test-data.json
    ```

    Make sure *test-data.json* file is in the direct directory of prompt command working directory.


## Python Script for Continuous Training

1. Data Management:

* Load data from an S3 bucket.

* Validate the structure of the loaded data to ensure consistency and correctness.

1. Data Preprocessing:

    Process the data by applying transformations and encodings. These steps should align with the preprocessing actions you took during the initial model training.

1. Model Training & Evaluation:

* Train an XGBoost machine learning model using the preprocessed data.

* Evaluate the model's performance to ensure it meets the expected standards.

1. Logging & Monitoring:

    Integrate AWS CloudWatch to log messages and important data. This will help in actively monitoring the continuous training process and troubleshooting any issues.

1. Model Deployment Criteria:

    Set a threshold for the model's R2 Score. If the model's performance meets or exceeds this threshold, then the trained model artifact will be saved and deployed to S3 for future use.

## Create CloudWatch Log group


## Schedule Continuous Training (Cron job)

Set up a cronjob to ensure the model is continuously trained at 9am UTC time everyday.

Refer to the bash script [ec2-cron-job.sh](ec2-cron-job.sh) for crontab commands.

P.S. Cron job V.S. Batch job
| Cron job | Batch job |
| -------- | --------- |
| A time managed script / command | a script that runs |
