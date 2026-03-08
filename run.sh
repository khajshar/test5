#!/bin/bash
export AZURE_ENDPOINT="${AZURE_ENDPOINT:-https://khajjj96-5663-resource.cognitiveservices.azure.com/openai/deployments/model-router/chat/completions?api-version=2025-01-01-preview}"
export AZURE_KEY="${AZURE_KEY:-}"
exec python3 azure_claude.py --listen-port 9000
