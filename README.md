# Develop and deploy an XGBoost House Rentability Forecasting model using AWS EC2 and API Gateway

<center><img src='https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fi%2F5p94qxqe2w9vzchif3ll.jpg' alt='' width='400' align='center'>
</center>

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
nohup python3 flask_app/main.py > model_serving_logs.txt 2>&1 &
```

