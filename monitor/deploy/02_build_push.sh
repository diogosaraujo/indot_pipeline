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

echo "Pushed ${ECR_URI}:${IMAGE_TAG}"
