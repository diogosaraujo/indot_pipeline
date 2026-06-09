@echo off
setlocal enabledelayedexpansion

set REGION=us-east-1
set SG_NAME=indot-pipeline-sg

echo Fetching your current public IP...
for /f %%i in ('curl -s https://api4.ipify.org') do set MY_IP=%%i
echo Your IP: %MY_IP%

echo Looking up security group...
for /f %%i in ('aws ec2 describe-security-groups --region %REGION% --filters "Name=group-name,Values=%SG_NAME%" --query "SecurityGroups[0].GroupId" --output text') do set SG_ID=%%i
echo Security Group ID: %SG_ID%

echo Revoking existing SSH rules...
for /f %%i in ('aws ec2 describe-security-groups --region %REGION% --group-ids %SG_ID% --query "SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp" --output text') do (
    aws ec2 revoke-security-group-ingress --group-id %SG_ID% --protocol tcp --port 22 --cidr %%i --region %REGION% >nul 2>&1
)
for /f %%i in ('aws ec2 describe-security-groups --region %REGION% --group-ids %SG_ID% --query "SecurityGroups[0].IpPermissions[?FromPort==`22`].Ipv6Ranges[].CidrIpv6" --output text') do (
    aws ec2 revoke-security-group-ingress --group-id %SG_ID% --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":22,\"ToPort\":22,\"Ipv6Ranges\":[{\"CidrIpv6\":\"%%i\"}]}]" --region %REGION% >nul 2>&1
)

echo Adding SSH rule for %MY_IP%...
echo %MY_IP% | findstr /C:":" >nul
if %ERRORLEVEL% EQU 0 (
    aws ec2 authorize-security-group-ingress --group-id %SG_ID% --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":22,\"ToPort\":22,\"Ipv6Ranges\":[{\"CidrIpv6\":\"%MY_IP%/128\"}]}]" --region %REGION%
) else (
    aws ec2 authorize-security-group-ingress --group-id %SG_ID% --protocol tcp --port 22 --cidr %MY_IP%/32 --region %REGION%
)

echo.
echo Done. You can now SSH from this machine.
