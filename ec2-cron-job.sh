
# SSH into your EC2 instance.
ssh -i "rentfore-keypair.pem" ubuntu@public-ipv4-dns

# Write the Cron Job to a File:
# Use echo to write the cron job to a new file. For instance, if your Python script path is /path/to/your_script.py, you can use:
# 0 9 * * * (Time Year Month Weekday)
echo "0 9 * * * /ds/bin/python model-training/ec2-continuous-training.py >> continuous_model_training_logs.log 2>&1" > model_training_cron_schedule

# Append the New Cron Job to the Current User's Crontab:
# Use the crontab command to append the new cron job:
crontab model_training_cron_schedule

# Ensure the cron job has been added, you can list the current user's crontab:
crontab -l

# Check the output logs after the Python script has been executed automatically as per the defined cron schedule
head continuous_model_training_logs.log
tail continuous_model_training_logs.log