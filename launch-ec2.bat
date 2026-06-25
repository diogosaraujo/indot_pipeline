@echo off
setlocal enabledelayedexpansion

:: --- CONFIGURATION ------------------------------------------------
set REGION=us-east-1
set INSTANCE_TYPE=m5.2xlarge
set IAM_ROLE=EC2-INDOT-Pipeline
set KEY_NAME=indot-pipeline-key
set SG_NAME=indot-pipeline-sg
:: ------------------------------------------------------------------

:: Step 1: Get your public IP
echo Fetching your public IP...
for /f %%i in ('curl -s https://ifconfig.me/ip') do set MY_IP=%%i
echo Your IP: %MY_IP%

:: Step 2: Get latest Ubuntu 24.04 LTS AMI
echo Finding latest Ubuntu 24.04 LTS AMI...
for /f %%i in ('aws ec2 describe-images --region %REGION% --owners 099720109477 --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text') do set AMI_ID=%%i
echo AMI ID: %AMI_ID%

:: Step 3: Check if security group already exists, reuse it if so
echo Checking for existing security group...
for /f %%i in ('aws ec2 describe-security-groups --region %REGION% --filters "Name=group-name,Values=%SG_NAME%" --query "SecurityGroups[0].GroupId" --output text 2^>nul') do set SG_ID=%%i

if "%SG_ID%"=="None" set SG_ID=
if "%SG_ID%"=="" (
    echo Security group not found, creating it...
    for /f %%i in ('aws ec2 create-security-group --group-name %SG_NAME% --description "INDOT Pipeline - SSH from my IP only" --region %REGION% --query "GroupId" --output text') do set SG_ID=%%i
    echo Created Security Group ID: %SG_ID%

    :: Add SSH inbound rule only when creating a new group
    echo Adding SSH inbound rule...
    aws ec2 authorize-security-group-ingress --group-id %SG_ID% --protocol tcp --port 22 --cidr %MY_IP%/32 --region %REGION%
) else (
    echo Reusing existing Security Group ID: %SG_ID%
)

:: Step 4: Launch the EC2 instance
echo Launching EC2 instance...
for /f %%i in ('aws ec2 run-instances --region %REGION% --image-id %AMI_ID% --instance-type %INSTANCE_TYPE% --key-name %KEY_NAME% --security-group-ids %SG_ID% --iam-instance-profile "Name=%IAM_ROLE%" --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":200,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=indot-pipeline}]" --query "Instances[0].InstanceId" --output text') do set INSTANCE_ID=%%i
echo Instance launched: %INSTANCE_ID%

:: Step 5: Wait for instance to be running
echo Waiting for instance to reach running state...
aws ec2 wait instance-running --instance-ids %INSTANCE_ID% --region %REGION%

:: Step 6: Get public IP
for /f %%i in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --region %REGION% --query "Reservations[0].Instances[0].PublicIpAddress" --output text') do set PUBLIC_IP=%%i

echo.
echo =========================================
echo  Instance is RUNNING!
echo  Instance ID : %INSTANCE_ID%
echo  Public IP   : %PUBLIC_IP%
echo  Connect with:
echo  ssh -i C:\Users\daraujo\Downloads\indot-pipeline-key.pem ubuntu@%PUBLIC_IP%
echo =========================================