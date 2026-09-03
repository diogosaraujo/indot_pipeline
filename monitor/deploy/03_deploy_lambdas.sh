#!/usr/bin/env bash
# Create or update the poller + alerter Lambda functions from the pushed image.
# Same image; each function overrides the CMD (handler) via ImageConfig.Command.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

# Prefer the digest 02 just pushed. A tag is resolved by Lambda at update time,
# and ECR propagation lags the push, so deploying by tag can silently pin an
# older image on whichever function updates first.
if [[ -f .last_image ]]; then          # CWD is the deploy dir (cd at the top)
  ECR_URI="$(cat .last_image)"
  echo "Deploying by digest: ${ECR_URI##*@}"
else
  ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
  echo "WARNING: no deploy/.last_image — falling back to :${IMAGE_TAG}."
  echo "         Verify both functions resolve to the same digest afterwards."
fi
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# ── recipient pre-flight ─────────────────────────────────────────────────────
# In the SES sandbox an unverified recipient does not merely miss its own mail:
# SendRawEmail rejects the WHOLE message, so one unverified address silently
# takes down alerting for everyone on the list. Deploying that config is worse
# than not adding the recipient at all, so refuse to.
if [[ "$(aws sesv2 get-account --region "$AWS_REGION" \
          --query ProductionAccessEnabled --output text)" != "True" ]]; then
  IFS=',' read -ra _R <<< "$MONITOR_ALERT_RECIPIENTS"
  _bad=()
  for r in "${_R[@]}" "$MONITOR_ALERT_SENDER"; do
    st="$(aws ses get-identity-verification-attributes --identities "$r" \
          --region "$AWS_REGION" \
          --query "VerificationAttributes.\"${r}\".VerificationStatus" \
          --output text 2>/dev/null || echo None)"
    [[ "$st" == "Success" ]] || _bad+=("$r ($st)")
  done
  if (( ${#_bad[@]} )); then
    echo "REFUSING TO DEPLOY — SES is in the sandbox and these identities are not verified:"
    printf '    %s\n' "${_bad[@]}"
    echo
    echo "  Each must click the AWS verification link in their own inbox."
    echo "  Re-send with:  aws ses verify-email-identity --email-address <addr> --region $AWS_REGION"
    echo "  Or remove the restriction entirely by requesting SES production access."
    exit 1
  fi
  echo "SES sandbox: all ${#_R[@]} recipient(s) + sender verified."
fi

COMMON_ENV="MONITOR_BUCKET=${MONITOR_BUCKET},MONITOR_PREFIX=${MONITOR_PREFIX}"
# The poller also needs the SES identities: it sends the source-staleness notice
# itself rather than routing a plain-text warning through the PDF alerter.
POLLER_ENV="{${COMMON_ENV},MONITOR_ALERTER_FUNCTION=${ALERTER_FN},MONITOR_FIRE_RP_SCOUR=${MONITOR_FIRE_RP_SCOUR},MONITOR_FIRE_RP_OTHER=${MONITOR_FIRE_RP_OTHER},MONITOR_ALERT_SENDER=${MONITOR_ALERT_SENDER},MONITOR_ALERT_RECIPIENTS=${MONITOR_ALERT_RECIPIENTS},MONITOR_STALE_WARN_HOURS=${MONITOR_STALE_WARN_HOURS:-3}}"
# The alerter also renders the daily summary ({"daily": true}), so it needs the
# window's timezone — the report is built for the previous LOCAL calendar day.
ALERTER_ENV="{${COMMON_ENV},MONITOR_ALERT_SENDER=${MONITOR_ALERT_SENDER},MONITOR_ALERT_RECIPIENTS=${MONITOR_ALERT_RECIPIENTS},MONITOR_DAILY_TZ=${MONITOR_DAILY_TZ:-America/New_York},MONITOR_DAILY_SEND_HOUR=${MONITOR_DAILY_SEND_HOUR:-6}}"

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
