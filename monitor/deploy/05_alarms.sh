#!/usr/bin/env bash
# CloudWatch alarms on the monitor itself.
#
# The staleness check watches the INPUTS. Nothing watched whether alerts were
# actually delivered, and a 95% alerter failure rate ran unnoticed for five days
# because a healthy poller and a quiet river look identical from outside. These
# alarms close that gap:
#
#   alerter-errors   any alerter failure          -> an alert was built and not sent
#   poller-errors    any poller failure           -> detection itself is broken
#   poller-silent    no invocations in 3 hours    -> the schedule stopped firing
#
# poller-silent treats MISSING data as breaching on purpose: if EventBridge is
# disabled or the function is deleted, the metric stops being published at all,
# which is precisely the case that must page someone.
#
# Notifications go through SNS (CloudWatch cannot email directly). The email
# subscription must be CONFIRMED from the inbox before anything is delivered.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

TOPIC_NAME="${ALARM_TOPIC:-indot-bridge-monitor-alarms}"
TOPIC_ARN="$(aws sns create-topic --name "$TOPIC_NAME" --region "$AWS_REGION" \
  --query TopicArn --output text)"
echo "Topic: $TOPIC_ARN"

IFS=',' read -ra RCPTS <<< "$MONITOR_ALERT_RECIPIENTS"
for r in "${RCPTS[@]}"; do
  existing="$(aws sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN" \
    --region "$AWS_REGION" --query "Subscriptions[?Endpoint=='${r}'].SubscriptionArn" \
    --output text 2>/dev/null || true)"
  if [[ -n "$existing" && "$existing" != "None" ]]; then
    echo "  $r already subscribed (${existing##*:})"
  else
    aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email --notification-endpoint "$r" \
      --region "$AWS_REGION" >/dev/null
    echo "  $r subscribed — CONFIRM the link in that inbox or nothing is delivered"
  fi
done

alarm () {  # name description metric fn stat period threshold operator missing
  aws cloudwatch put-metric-alarm \
    --alarm-name "$1" --alarm-description "$2" \
    --namespace AWS/Lambda --metric-name "$3" \
    --dimensions "Name=FunctionName,Value=$4" \
    --statistic "$5" --period "$6" --threshold "$7" \
    --comparison-operator "$8" --treat-missing-data "$9" \
    --evaluation-periods 1 \
    --alarm-actions "$TOPIC_ARN" --ok-actions "$TOPIC_ARN" \
    --region "$AWS_REGION"
  echo "  alarm: $1"
}

alarm "indot-bridge-alerter-errors" \
      "Alerter failed — an alert was built but not delivered" \
      Errors "$ALERTER_FN" Sum 300 0 GreaterThanThreshold notBreaching

alarm "indot-bridge-poller-errors" \
      "Poller failed — bridge conditions are not being evaluated" \
      Errors "$POLLER_FN" Sum 300 0 GreaterThanThreshold notBreaching

alarm "indot-bridge-poller-silent" \
      "Poller has not run in 3 hours — the hourly schedule may have stopped" \
      Invocations "$POLLER_FN" Sum 10800 1 LessThanThreshold breaching

echo
echo "Alarms in place. Check state with:"
echo "  aws cloudwatch describe-alarms --alarm-name-prefix indot-bridge --region $AWS_REGION \\"
echo "    --query 'MetricAlarms[].{Name:AlarmName,State:StateValue}' --output table"
