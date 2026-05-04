@echo off
setlocal enabledelayedexpansion

:: ─── CONFIGURATION ────────────────────────────────────────────────
set REGION=us-east-1
set BUCKET_NAME=indot-bridge-pipeline-YOUR-ID
:: Replace YOUR-ID with your initials or another unique suffix.
:: Bucket names must be globally unique across all AWS accounts.
:: Use only lowercase letters, numbers, and hyphens.
:: ──────────────────────────────────────────────────────────────────

:: Create the bucket
echo Creating S3 bucket: !BUCKET_NAME! in !REGION!...
aws s3api create-bucket --bucket !BUCKET_NAME! --region !REGION!
if errorlevel 1 (
    echo ERROR: Failed to create bucket. Check that the name is globally unique and you have s3:CreateBucket permission.
    exit /b 1
)

:: Block all public access
echo Blocking public access...
aws s3api put-public-access-block ^
    --bucket !BUCKET_NAME! ^
    --public-access-block-configuration ^
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo.
echo =========================================
echo  S3 bucket created!
echo  Bucket : !BUCKET_NAME!
echo  Region : !REGION!
echo.
echo  Next steps:
echo    1. Set BUCKET_NAME=!BUCKET_NAME! in create-iam-role.bat
echo    2. Run create-iam-role.bat
echo    3. Update config.yaml with this bucket name
echo =========================================
