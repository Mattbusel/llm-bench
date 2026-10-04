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
# Streamed replies end with a two-byte character that is deliberately split
# across two network writes, to exercise the client's stream decoding.
STREAM_REPLY = REPLY + " Café."


def simulate(model, prompt):
    base, jitter, out_tokens = PROFILES.get(model, DEFAULT)
    time.sleep(max(0.05, base + random.uniform(-jitter / 2, jitter)))
    prompt_tokens = max(1, len(prompt.split()) * 4 // 3)
    return prompt_tokens, out_tokens + random.randint(-6, 6)


def stream(handler, model, prompt, anthropic):
    """Send the reply as server-sent events: a pause before the first token
    (40% of the profile's latency), then the words spread over the rest."""
    base, jitter, out_tokens = PROFILES.get(model, DEFAULT)
    total = max(0.05, base + random.uniform(-jitter / 2, jitter))
    prompt_tokens = max(1, len(prompt.split()) * 4 // 3)
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.end_headers()

    def send(event, obj):
        line = (f"event: {event}\n" if anthropic else "") + "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        data = line.encode()
        if "é" in line:
            cut = data.index("é".encode()) + 1  # inside the two UTF-8 bytes
            handler.wfile.write(data[:cut]); handler.wfile.flush(); time.sleep(0.02)
            data = data[cut:]
        handler.wfile.write(data); handler.wfile.flush()

    if anthropic:
        send("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": prompt_tokens, "output_tokens": 1}}})
    time.sleep(total * 0.4)
    words = STREAM_REPLY.split(" ")
    for i, w in enumerate(words):
        piece = w if i == 0 else " " + w
        if anthropic:
            send("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}})
        else:
            send("chunk", {"choices": [{"index": 0, "delta": {"content": piece}}]})
        time.sleep(total * 0.6 / len(words))
    if anthropic:
        send("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": out_tokens}})
        send("message_stop", {"type": "message_stop"})
    else:
        send("chunk", {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": out_tokens, "total_tokens": prompt_tokens + out_tokens}})
        handler.wfile.write(b"data: [DONE]\n\n"); handler.wfile.flush()


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
        if req.get("stream") and self.path in ("/v1/chat/completions", "/v1/messages"):
            stream(self, model, prompt, self.path == "/v1/messages")
        elif self.path == "/v1/chat/completions":
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
