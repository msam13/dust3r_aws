#!/bin/bash

echo "Starting DUSt3R inference process..."

# Extract input parameters (assuming JSON input)
payload=$(cat /dev/stdin)
batch_id=$(echo $payload | jq -r '.batch_id')
photo_keys=$(echo $payload | jq -r '.photo_keys[]')

# Initialize AWS CLI with the specified region
aws configure set region us-east-1

# Download images from S3
mkdir -p /tmp/input_images
mkdir -p /tmp/output
for key in $photo_keys; do
    local_file="/tmp/input_images/$(basename $key)"
    aws s3 cp "s3://your-photo-bucket/$key" "$local_file"
done

# Run inference
python3 inference.py --input_dir /tmp/input_images --output_dir /tmp/output

# Upload results to S3
output_file="/tmp/output/model.obj"
aws s3 cp "$output_file" "s3://your-model-bucket/$batch_id/model.obj"

# Return the path of the uploaded model
echo "{\"generated_model_path\": \"s3://your-model-bucket/$batch_id/model.obj\"}"

