#!/usr/bin/env python3
"""azure_claude.py — Azure OpenAI to Anthropic API adapter.

Converts Anthropic /v1/messages requests to Azure OpenAI format.
Responds in Anthropic streaming format.

Usage:
    export AZURE_ENDPOINT="https://..."
    export AZURE_KEY="..."
    python3 azure_claude.py --listen-port 9000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class AzureClaudeAdapter(BaseHTTPRequestHandler):
    """HTTP handler that adapts Anthropic requests to Azure OpenAI."""

    AZURE_ENDPOINT = ""
    AZURE_KEY = ""

    @classmethod
    def set_credentials(cls, endpoint: str, key: str):
        cls.AZURE_ENDPOINT = endpoint
        cls.AZURE_KEY = key

    def do_POST(self) -> None:
        """Handle POST /v1/messages."""
        if self.path != "/v1/messages":
            self.send_error(404)
            return

        if not self.AZURE_ENDPOINT or not self.AZURE_KEY:
            self.send_error(500, "Azure credentials not configured")
            return

        try:
            content_len = int(self.headers.get("content-length", 0))
            req_body = json.loads(self.rfile.read(content_len))
        except Exception as e:
            self.send_error(400, str(e))
            return

        try:
            # Convert request
            azure_payload = self._convert_request(req_body)
            # Call Azure
            azure_response = self._call_azure(azure_payload)
            # Convert response back to Anthropic format
            response_lines = self._convert_response(azure_response)

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            for line in response_lines:
                self.wfile.write(line)

        except Exception as e:
            print(f"[error] {e}", flush=True)
            self.send_error(500, str(e))

    def _convert_request(self, anthropic_req: dict) -> dict:
        """Convert Anthropic /v1/messages to Azure OpenAI format."""
        messages = []

        # Handle system prompt
        system = anthropic_req.get("system")
        if system:
            if isinstance(system, str):
                messages.append({"role": "user", "content": system})
            elif isinstance(system, list):
                text = "".join(
                    block.get("text", "")
                    for block in system
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if text:
                    messages.append({"role": "user", "content": text})

        # Convert messages
        for msg in anthropic_req.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Flatten content blocks
            if isinstance(content, list):
                text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                content = text

            messages.append({"role": role, "content": content})

        return {
            "model": anthropic_req.get("model", "gpt-4"),
            "messages": messages,
            "max_tokens": anthropic_req.get("max_tokens", 4096),
            "temperature": anthropic_req.get("temperature", 1.0),
        }

    def _call_azure(self, payload: dict) -> dict:
        """Call Azure OpenAI endpoint."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.AZURE_KEY,
        }

        req = urllib.request.Request(
            self.AZURE_ENDPOINT,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Azure {e.code}: {e.read().decode()}")
        except Exception as e:
            raise RuntimeError(f"Azure call failed: {e}")

    def _convert_response(self, azure_resp: dict) -> list[bytes]:
        """Convert Azure response to Anthropic streaming format."""
        lines = []

        # Extract content from first choice
        choice = azure_resp.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")

        # Message start event
        lines.append(
            json.dumps({
                "type": "message_start",
                "message": {
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": azure_resp.get("model", "gpt-4"),
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }).encode() + b"\n"
        )

        # Content block start
        lines.append(
            json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }).encode() + b"\n"
        )

        # Content delta
        lines.append(
            json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content},
            }).encode() + b"\n"
        )

        # Content block stop
        lines.append(
            json.dumps({
                "type": "content_block_stop",
                "index": 0,
            }).encode() + b"\n"
        )

        # Message delta
        lines.append(
            json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": len(content.split())},
            }).encode() + b"\n"
        )

        # Message stop
        lines.append(
            json.dumps({
                "type": "message_stop",
            }).encode() + b"\n"
        )

        return lines

    def log_message(self, format: str, *args: Any) -> None:
        """Log with flush."""
        print(f"[{self.client_address[0]}] {format % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Azure Claude adapter")
    parser.add_argument("--listen-port", type=int, default=9000)
    parser.add_argument("--listen-host", default="127.0.0.1")
    args = parser.parse_args()

    endpoint = os.environ.get("AZURE_ENDPOINT", "")
    key = os.environ.get("AZURE_KEY", "")

    if not endpoint or not key:
        print("[error] AZURE_ENDPOINT and AZURE_KEY required", flush=True)
        sys.exit(1)

    AzureClaudeAdapter.set_credentials(endpoint, key)

    server = ThreadingHTTPServer(
        (args.listen_host, args.listen_port),
        AzureClaudeAdapter,
    )
    print(f"[info] Listening on {args.listen_host}:{args.listen_port}", flush=True)
    print(f"[info] Azure: {endpoint[:60]}...", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[info] Shutdown", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
