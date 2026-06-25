@echo off
setlocal enabledelayedexpansion

:: --- CONFIGURATION ------------------------------------------------
set ROLE_NAME=EC2-INDOT-Pipeline
set BUCKET_NAME=indot-bridge-pipeline-YOUR-ID
:: Set BUCKET_NAME to the same value used in create-s3-bucket.bat
:: ------------------------------------------------------------------

set TRUST_FILE=%TEMP%\indot-trust-policy.json
set POLICY_FILE=%TEMP%\indot-s3-policy.json

:: Write EC2 trust policy to a temp file
echo Writing trust policy...
(
echo {
echo   "Version": "2012-10-17",
echo   "Statement": [
echo     {
echo       "Effect": "Allow",
echo       "Principal": { "Service": "ec2.amazonaws.com" },
echo       "Action": "sts:AssumeRole"
echo     }
echo   ]
echo }
) > "!TRUST_FILE!"

:: Create the IAM role
echo Creating IAM role: !ROLE_NAME!...
aws iam create-role ^
    --role-name !ROLE_NAME! ^
    --assume-role-policy-document file://"!TRUST_FILE!" ^
    --description "EC2 role for INDOT pipeline - scoped S3 read/write access"
if errorlevel 1 (
    echo WARNING: Role may already exist. Continuing to policy attachment...
)

:: Write scoped inline policy to a temp file
:: Grants full access to the pipeline output bucket and read-only access to the public NOAA MRMS bucket
echo Writing S3 access policy...
(
echo {
echo   "Version": "2012-10-17",
echo   "Statement": [
echo     {
echo       "Sid": "PipelineBucketAccess",
echo       "Effect": "Allow",
echo       "Action": [
echo         "s3:GetObject",
echo         "s3:PutObject",
echo         "s3:DeleteObject",
echo         "s3:ListBucket",
echo         "s3:GetBucketLocation"
echo       ],
echo       "Resource": [
echo         "arn:aws:s3:::!BUCKET_NAME!",
echo         "arn:aws:s3:::!BUCKET_NAME!/*"
echo       ]
echo     },
echo     {
echo       "Sid": "NOAAMRMSReadOnly",
echo       "Effect": "Allow",
echo       "Action": [
echo         "s3:GetObject",
echo         "s3:ListBucket"
echo       ],
echo       "Resource": [
echo         "arn:aws:s3:::noaa-mrms-pds",
echo         "arn:aws:s3:::noaa-mrms-pds/*"
echo       ]
echo     }
echo   ]
echo }
) > "!POLICY_FILE!"

:: Attach inline policy to the role
echo Attaching S3 policy to role...
aws iam put-role-policy ^
    --role-name !ROLE_NAME! ^
    --policy-name INDOT-S3-Access ^
    --policy-document file://"!POLICY_FILE!"
if errorlevel 1 (
    echo ERROR: Failed to attach policy. Aborting.
    del "!TRUST_FILE!" 2>nul
    del "!POLICY_FILE!" 2>nul
    exit /b 1
)

:: Create the instance profile (must share the role name so launch-ec2.bat finds it)
echo Creating instance profile...
aws iam create-instance-profile --instance-profile-name !ROLE_NAME!
if errorlevel 1 (
    echo WARNING: Instance profile may already exist. Continuing...
)

:: Add the role to the instance profile
echo Adding role to instance profile...
aws iam add-role-to-instance-profile ^
    --instance-profile-name !ROLE_NAME! ^
    --role-name !ROLE_NAME!
if errorlevel 1 (
    echo WARNING: Role may already be attached to instance profile. Continuing...
)

:: Clean up temp files
del "!TRUST_FILE!" 2>nul
del "!POLICY_FILE!" 2>nul

echo.
echo =========================================
echo  IAM role and instance profile created!
echo  Role / profile : !ROLE_NAME!
echo  Scoped bucket  : !BUCKET_NAME!
echo.
echo  IAM changes take ~15 seconds to propagate.
echo  Wait before running launch-ec2.bat.
echo =========================================
