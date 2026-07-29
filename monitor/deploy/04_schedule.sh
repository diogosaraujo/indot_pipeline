#!/usr/bin/env bash
# Hourly EventBridge rule that invokes the poller (default :55 past each hour,
# after MRMS Pass2 / NWM analysis publish, ~1 h latency).
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

RULE=indot-bridge-poller-hourly
POLLER_ARN="arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${POLLER_FN}"

aws events put-rule --name "$RULE" --schedule-expression "$SCHEDULE_CRON" \
  --state ENABLED --region "$AWS_REGION" >/dev/null

aws lambda add-permission --function-name "$POLLER_FN" \
  --statement-id "${RULE}-invoke" --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${AWS_REGION}:${ACCOUNT_ID}:rule/${RULE}" \
  --region "$AWS_REGION" 2>/dev/null || echo "invoke permission already present"

aws events put-targets --rule "$RULE" --region "$AWS_REGION" \
  --targets "Id=poller,Arn=${POLLER_ARN}" >/dev/null

echo "Scheduled ${POLLER_FN} on ${RULE} (${SCHEDULE_CRON})."
echo "Test now:  aws lambda invoke --function-name ${POLLER_FN} --region ${AWS_REGION} /tmp/out.json && cat /tmp/out.json"
