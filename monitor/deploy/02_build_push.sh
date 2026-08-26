#!/usr/bin/env bash
# Build the container image and push it to ECR. Run from anywhere; build context
# is the repo root (two levels up) so the Dockerfile can COPY monitor/.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env
REPO_ROOT="$(cd ../.. && pwd)"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build --platform linux/amd64 -f "${REPO_ROOT}/monitor/Dockerfile" \
  -t "${ECR_URI}:${IMAGE_TAG}" "${REPO_ROOT}"
# Capture the digest from docker's own push output. Reading it back from ECR
# would need ecr:DescribeImages, which the EC2 role deliberately does not have,
# and this is the authoritative digest of what was just pushed anyway.
PUSH_LOG="$(mktemp)"
docker push "${ECR_URI}:${IMAGE_TAG}" | tee "$PUSH_LOG"
DIGEST="$(awk '/digest: sha256:/ {print $3}' "$PUSH_LOG" | tail -1)"
rm -f "$PUSH_LOG"

if [[ -z "$DIGEST" ]]; then
  echo "WARNING: could not parse a digest from the push output."
  echo "         03 will fall back to the :${IMAGE_TAG} tag — verify afterwards that"
  echo "         both functions resolve to the same image."
else
  # CWD is already the deploy dir (the cd at the top), so this is relative to it.
  echo "${ECR_URI}@${DIGEST}" > .last_image
  echo "Digest ${DIGEST}  (written to deploy/.last_image for 03 to use)"
fi

echo "Pushed ${ECR_URI}:${IMAGE_TAG}"
