@echo off
setlocal enabledelayedexpansion

:: --- CONFIGURATION ------------------------------------------------
set REGION=us-east-1
set TARGET_TYPE=m5.2xlarge
set INSTANCE_NAME=indot-pipeline
set KEY_NAME=indot-pipeline-key
:: ------------------------------------------------------------------

echo Looking up instance ID for "%INSTANCE_NAME%"...
for /f %%i in ('aws ec2 describe-instances --region %REGION% --filters "Name=tag:Name,Values=%INSTANCE_NAME%" "Name=instance-state-name,Values=running,stopped" --query "Reservations[0].Instances[0].InstanceId" --output text') do set INSTANCE_ID=%%i

if "%INSTANCE_ID%"=="" (
    echo ERROR: No running or stopped instance named "%INSTANCE_NAME%" found.
    exit /b 1
)
echo Instance ID: %INSTANCE_ID%

echo Stopping instance...
aws ec2 stop-instances --instance-ids %INSTANCE_ID% --region %REGION% > nul
echo Waiting for instance to stop...
aws ec2 wait instance-stopped --instance-ids %INSTANCE_ID% --region %REGION%
echo Instance stopped.

echo Restoring instance type to %TARGET_TYPE% (32 GB RAM)...
aws ec2 modify-instance-attribute --instance-id %INSTANCE_ID% --instance-type "{\"Value\":\"%TARGET_TYPE%\"}" --region %REGION%

echo Starting instance...
aws ec2 start-instances --instance-ids %INSTANCE_ID% --region %REGION% > nul
echo Waiting for instance to reach running state...
aws ec2 wait instance-running --instance-ids %INSTANCE_ID% --region %REGION%

for /f %%i in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --region %REGION% --query "Reservations[0].Instances[0].PublicIpAddress" --output text') do set PUBLIC_IP=%%i

echo.
echo =========================================
echo  Instance restored to %TARGET_TYPE% (32 GB)
echo  Instance ID : %INSTANCE_ID%
echo  Public IP   : %PUBLIC_IP%
echo  Connect with:
echo  ssh -i C:\Users\daraujo\Downloads\indot-pipeline-key.pem ubuntu@%PUBLIC_IP%
echo =========================================
