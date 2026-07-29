#!/usr/bin/env bash
# One-time SES setup. In the SES *sandbox* both sender and each recipient must be
# verified; to email arbitrary recipients, request production access (link below).
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

echo "Verifying sender: $MONITOR_ALERT_SENDER"
aws ses verify-email-identity --email-address "$MONITOR_ALERT_SENDER" --region "$AWS_REGION"

IFS=',' read -ra RCPTS <<< "$MONITOR_ALERT_RECIPIENTS"
for r in "${RCPTS[@]}"; do
  echo "Verifying recipient (sandbox only): $r"
  aws ses verify-email-identity --email-address "$r" --region "$AWS_REGION"
done

cat <<'NOTE'

Check each inbox and click the AWS verification link.
To send to unverified recipients, request production access:
  https://console.aws.amazon.com/ses/  ->  Account dashboard  ->  Request production access
Verifying a whole domain (DKIM) instead of individual addresses is recommended
for a real deployment.
NOTE
