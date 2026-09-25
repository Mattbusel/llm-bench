"""A tiny local stand-in for the OpenAI and Anthropic APIs.

It exists so you can try llm-bench (and so the README demo could be recorded)
without API keys. Nothing here talks to a real model: every reply is canned
text, and the latency is simulated with a per-model delay plus jitter.

    python scripts/mock_server.py            # listens on http://127.0.0.1:8787
    llm-bench run --openai-base-url http://127.0.0.1:8787 \
                  --anthropic-base-url http://127.0.0.1:8787 \
                  --models gpt-4o-mini,gpt-4o,claude-haiku-4-5 --prompts "Say hi"

Standard library only.
"""

import json
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787

# Simulated (base latency in seconds, jitter in seconds, reply length in tokens).
PROFILES = {
    "gpt-4o-mini": (0.32, 0.12, 38),
    "gpt-4o": (0.62, 0.25, 52),
    "claude-haiku-4-5": (0.41, 0.15, 44),
}
DEFAULT = (0.5, 0.2, 40)
REPLY = "This is a canned reply from the local mock server, not a real model."


def simulate(model, prompt):
    base, jitter, out_tokens = PROFILES.get(model, DEFAULT)
    time.sleep(max(0.05, base + random.uniform(-jitter / 2, jitter)))
    prompt_tokens = max(1, len(prompt.split()) * 4 // 3)
    return prompt_tokens, out_tokens + random.randint(-6, 6)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        model = req.get("model", "unknown")
        prompt = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in req.get("messages", [])
        )
        if self.path == "/v1/chat/completions":
            p, c = simulate(model, prompt)
            self._json(200, {
                "id": "mock", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": REPLY}}],
                "usage": {"prompt_tokens": p, "completion_tokens": c,
                          "total_tokens": p + c},
            })
        elif self.path == "/v1/messages":
            p, c = simulate(model, prompt)
            self._json(200, {
                "id": "mock", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": REPLY}],
                "usage": {"input_tokens": p, "output_tokens": c},
            })
        else:
            self._json(404, {"error": f"mock server has no route {self.path}"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"mock LLM API on http://127.0.0.1:{PORT} (canned replies, simulated latency)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
