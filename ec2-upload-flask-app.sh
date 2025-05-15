# Change the permissions of the pem file
chmod 400 file-name.pem

# Connect to EC2 instance and create a folder:flask_app at the root of EC2 file system
ssh -i "file-name.pem" ubuntu@public-ipv4-dns.com
mkdir flask_app
exit

scp -i "udemy-ds-keypair.pem" flask-script/main.py ubuntu@public-ipv4-dns.com:flask_app

# Connect to EC2 instance and run the flask app: main.py
ssh -i "file-name.pem" ubuntu@public-ipv4-dns.com
python3 flask_app/main.py

# Run the flask app in the background
nohup python3 flask_app/main.py > model_serving_logs.txt 2>&1 &