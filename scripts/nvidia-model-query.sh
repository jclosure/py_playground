#!/bin/bash
# NVIDIA Model Query Script
# Queries NVIDIA NIM API for available models

set -e

API_KEY="${NVIDIA_API_KEY:-}"
if [ -z "$API_KEY" ]; then
    echo "Warning: NVIDIA_API_KEY not set"
fi

echo "Querying NVIDIA NIM models..."
curl -s "https://integrate.api.nvidia.com/v1/models" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" 2>/dev/null || echo "API query failed"
