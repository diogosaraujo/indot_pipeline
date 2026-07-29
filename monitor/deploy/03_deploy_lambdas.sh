#!/usr/bin/env bash
# Create or update the poller + alerter Lambda functions from the pushed image.
# Same image; each function overrides the CMD (handler) via ImageConfig.Command.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

COMMON_ENV="MONITOR_BUCKET=${MONITOR_BUCKET},MONITOR_PREFIX=${MONITOR_PREFIX}"
POLLER_ENV="{${COMMON_ENV},MONITOR_ALERTER_FUNCTION=${ALERTER_FN},MONITOR_FIRE_RP_SCOUR=${MONITOR_FIRE_RP_SCOUR},MONITOR_FIRE_RP_OTHER=${MONITOR_FIRE_RP_OTHER}}"
ALERTER_ENV="{${COMMON_ENV},MONITOR_ALERT_SENDER=${MONITOR_ALERT_SENDER},MONITOR_ALERT_RECIPIENTS=${MONITOR_ALERT_RECIPIENTS}}"

deploy () {   # name handler memory timeout env
  local NAME="$1" HANDLER="$2" MEM="$3" TMO="$4" ENV="$5"
  if aws lambda get-function --function-name "$NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Updating $NAME ..."
    aws lambda update-function-code --function-name "$NAME" --image-uri "$ECR_URI" \
      --region "$AWS_REGION" >/dev/null
    aws lambda wait function-updated --function-name "$NAME" --region "$AWS_REGION"
    aws lambda update-function-configuration --function-name "$NAME" \
      --memory-size "$MEM" --timeout "$TMO" --role "$ROLE_ARN" \
      --image-config "Command=[$HANDLER]" \
      --environment "Variables=$ENV" --region "$AWS_REGION" >/dev/null
  else
    echo "Creating $NAME ..."
    aws lambda create-function --function-name "$NAME" --package-type Image \
      --code ImageUri="$ECR_URI" --role "$ROLE_ARN" \
      --image-config "Command=[$HANDLER]" \
      --memory-size "$MEM" --timeout "$TMO" --architectures x86_64 \
      --environment "Variables=$ENV" --region "$AWS_REGION" >/dev/null
  fi
  aws lambda wait function-updated --function-name "$NAME" --region "$AWS_REGION"
  echo "  $NAME ready."
}

# Alerter first so the poller's InvokeFunction target exists.
deploy "$ALERTER_FN" "lambda_alerter.handler" "$ALERTER_MEMORY" "$ALERTER_TIMEOUT" "$ALERTER_ENV"
deploy "$POLLER_FN"  "lambda_poller.handler"  "$POLLER_MEMORY"  "$POLLER_TIMEOUT"  "$POLLER_ENV"
