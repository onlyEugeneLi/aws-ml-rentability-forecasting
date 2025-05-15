# Change the permissions of the pem file
chmod 400 "ds-keypair.pem"

# Connect to Ec2 and create a folder : db_files inside the EC2 Instance
ssh -i "keypair.pem" ubuntu@{ec2-public-ipv4-dns}

# Install all the necessary Data science modules including flask
sudo apt update
sudo apt install python3-pip
pip install xgboost==1.6.2 pandas==1.5.2 joblib==1.2.0 scikit-learn==1.1.1 matplotlib==3.5.2 boto3 fsspec s3fs==2023.9.2 seaborn
sudo apt install python3-flask


# Install jupyterlab and notebook
pip install jupyterlab
pip install jupyter notebook
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
source ~/.bashrc

# Set the password for jupyterlab 
jupyter notebook password

# Start virtual environment

source ds/bin/activate
# or
source ml-lab/bin/activate

# Initiate jupyterlab 
jupyter-lab --ip 0.0.0.0 --no-browser --allow-root
