#!/usr/bin/env bash
# Daily summary schedule — one email every morning, quiet days included.
#
# Uses EventBridge SCHEDULER, not an EventBridge rule, because only Scheduler
# honours a timezone. A classic rule is UTC-only, so "06:00 Eastern" would have
# to be pinned at 10:00 UTC and would silently become 05:00 local every winter.
# The report is meant to land before the working day starts; a schedule that
# drifts an hour twice a year is a schedule nobody trusts.
#
# The target is the ALERTER with {"daily": true} — same image, same SES identity,
# same size-guarded send path (which is what stopped the 10 MB SES failures), so
# there is no third function to keep in step.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

SCHED_NAME="${DAILY_SCHEDULE_NAME:-indot-bridge-daily-summary}"
SCHED_ROLE="${DAILY_SCHEDULE_ROLE:-indot-bridge-scheduler-role}"
HOUR="${MONITOR_DAILY_SEND_HOUR:-6}"
TZ_NAME="${MONITOR_DAILY_TZ:-America/New_York}"
ALERTER_ARN="arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${ALERTER_FN}"

# ── role Scheduler assumes to invoke the alerter ─────────────────────────────
if ! aws iam get-role --role-name "$SCHED_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$SCHED_ROLE" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow",
                    "Principal":{"Service":"scheduler.amazonaws.com"},
                    "Action":"sts:AssumeRole"}]}' >/dev/null
  echo "Created role $SCHED_ROLE"
  sleep 10                      # IAM propagation before Scheduler validates it
else
  echo "Role $SCHED_ROLE already exists"
fi

aws iam put-role-policy --role-name "$SCHED_ROLE" --policy-name invoke-alerter \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",
                   \"Resource\":[\"${ALERTER_ARN}\",\"${ALERTER_ARN}:*\"]}]}"
ROLE_ARN="$(aws iam get-role --role-name "$SCHED_ROLE" --query Role.Arn --output text)"

# ── the schedule ─────────────────────────────────────────────────────────────
# FLEXIBLE_TIME_WINDOW OFF: the window is a fixed calendar day, so there is no
# value in letting AWS smear the start time.
ARGS=(--name "$SCHED_NAME"
      --schedule-expression "cron(0 ${HOUR} * * ? *)"
      --schedule-expression-timezone "$TZ_NAME"
      --flexible-time-window '{"Mode":"OFF"}'
      --target "{\"Arn\":\"${ALERTER_ARN}\",\"RoleArn\":\"${ROLE_ARN}\",
                 \"Input\":\"{\\\"daily\\\":true}\",
                 \"RetryPolicy\":{\"MaximumRetryAttempts\":2,
                                  \"MaximumEventAgeInSeconds\":3600}}"
      --description "INDOT bridge monitor — daily summary for the previous local day"
      --region "$AWS_REGION")

if aws scheduler get-schedule --name "$SCHED_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws scheduler update-schedule "${ARGS[@]}" >/dev/null
  echo "Updated schedule $SCHED_NAME"
else
  aws scheduler create-schedule "${ARGS[@]}" >/dev/null
  echo "Created schedule $SCHED_NAME"
fi

echo "Daily summary at ${HOUR}:00 ${TZ_NAME} -> ${ALERTER_FN} {\"daily\":true}"
echo
echo "Test now (renders yesterday and emails it):"
echo "  aws lambda invoke --function-name ${ALERTER_FN} --region ${AWS_REGION} \\"
echo "    --cli-read-timeout 0 --payload '{\"daily\":true}' /tmp/daily.json && cat /tmp/daily.json"
