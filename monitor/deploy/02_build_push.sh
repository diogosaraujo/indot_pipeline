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
docker push "${ECR_URI}:${IMAGE_TAG}"

# Resolve the tag to an immutable digest and hand it to 03. Lambda resolves a
# tag at update time, and ECR tag propagation lags the push by seconds — long
# enough that a deploy run immediately after a push once pinned the ALERTER to
# the previous image while the poller got the new one. Deploying by digest
# removes the race entirely.
DIGEST="$(aws ecr describe-images --repository-name "$ECR_REPO" --region "$AWS_REGION"   --image-ids imageTag="$IMAGE_TAG" --query 'imageDetails[0].imageDigest' --output text)"
echo "${ECR_URI}@${DIGEST}" > "$(dirname "$0")/.last_image"

echo "Pushed ${ECR_URI}:${IMAGE_TAG}"
echo "Digest ${DIGEST}  (written to deploy/.last_image for 03 to use)"
