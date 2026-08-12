#!/usr/bin/env bash
# Create the shared Lambda execution role for the monitor.
# NOAA source buckets are read anonymously (anon=True), so the role only needs:
#   own bucket R/W, SES send, invoke the alerter, and CloudWatch Logs.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document "$TRUST" \
  --description "INDOT bridge monitor Lambda role" 2>/dev/null \
  || echo "Role $ROLE_NAME already exists — continuing."

aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Written next to this script rather than /tmp, and referenced by a RELATIVE
# file:// path: a native Windows aws.exe under Git Bash resolves
# file:///tmp/... to C:\tmp\... and fails, while a relative path works on both.
POLICY_FILE=monitor-inline.json
trap 'rm -f "$POLICY_FILE"' EXIT

cat > "$POLICY_FILE" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PipelineBucketRW",
      "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket","s3:GetBucketLocation"],
      "Resource": ["arn:aws:s3:::${MONITOR_BUCKET}","arn:aws:s3:::${MONITOR_BUCKET}/*"]
    },
    {
      "Sid": "SendAlertEmail",
      "Effect": "Allow",
      "Action": ["ses:SendRawEmail","ses:SendEmail"],
      "Resource": "*"
    },
    {
      "Sid": "InvokeAlerter",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${ALERTER_FN}"
    }
  ]
}
JSON

aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name monitor-access --policy-document "file://${POLICY_FILE}"

echo "Role ready: arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "(IAM changes take ~15 s to propagate before deploying the functions.)"
