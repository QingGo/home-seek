"""
Home-Seek API Server — OpenAI-compatible API + interactive terminal mode.

Usage:
    python -m home_seek.api_server                    # API server (port 8000)
    python -m home_seek.api_server --interactive      # Terminal REPL
    python -m home_seek.api_server --interactive --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import os
import json
import time
import argparse
import uuid
import logging
from typing import AsyncGenerator

import torch

_logger = logging.getLogger("home-seek")
_log = _logger.info


class _TokenBuffer:
    """Buffer token IDs so the tokenizer can properly assemble
    multi-byte UTF-8 sequences from byte-level BPE tokens.

    The tokenizers library uses ``String::from_utf8_lossy`` under the hood,
    which replaces invalid byte sequences with U+FFFD (replacement character).
    When we decode a partial multi-byte sequence, the decoded text contains
    U+FFFD. As more bytes arrive the text can *shrink* (e.g. "��" → "お").
    This class tracks the first U+FFFD position and never emits text
    near unresolved byte sequences.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._ids: list[int] = []
        self._emitted: int = 0

    def reset(self):
        self._ids.clear()
        self._emitted = 0

    def add(self, tok_id: int) -> str:
        self._ids.append(tok_id)
        text = self.tokenizer.decode(self._ids, skip_special_tokens=True)

        first_bad = text.find('\ufffd')
        safe_end = first_bad if first_bad >= 0 else len(text)

        if safe_end <= self._emitted:
            return ""
        new_text = text[self._emitted:safe_end]
        self._emitted = safe_end
        return new_text

    def flush(self) -> str:
        """Emit any remaining text (call at end of generation)."""
        if not self._ids:
            return ""
        text = self.tokenizer.decode(self._ids, skip_special_tokens=True)
        if len(text) <= self._emitted:
            return ""
        new_text = text[self._emitted:]
        self._emitted = len(text)
        return new_text


from home_seek.inference_engine import HomeSeekInferenceEngine


# ── Statistics ──────────────────────────────────────────────────────────

class RequestStats:
    """Per-request performance tracking."""
    def __init__(self):
        self.encoding_tokens = 0       # prompt tokens
        self.encoding_time_s = 0.0     # prefill time
        self.generated_tokens = 0       # decode tokens
        self.decode_time_s = 0.0       # decode time
        self.ttft_s = 0.0              # time to first token

    @property
    def encoding_speed(self) -> float:
        return self.encoding_tokens / self.encoding_time_s if self.encoding_time_s > 0 else 0

    @property
    def decode_speed(self) -> float:
        return self.generated_tokens / self.decode_time_s if self.decode_time_s > 0 else 0

    def summary(self) -> str:
        return (
            f"prefill={self.encoding_tokens}tok/{self.encoding_time_s*1000:.0f}ms"
            f"({self.encoding_speed:.1f}t/s)  "
            f"decode={self.generated_tokens}tok/{self.decode_time_s*1000:.0f}ms"
            f"({self.decode_speed:.1f}t/s)"
        )

    def dict(self) -> dict:
        return {
            "encoding_tokens": self.encoding_tokens,
            "encoding_time_ms": round(self.encoding_time_s * 1000),
            "encoding_speed_tps": round(self.encoding_speed, 1),
            "ttft_ms": round(self.ttft_s * 1000),
            "generated_tokens": self.generated_tokens,
            "decode_time_ms": round(self.decode_time_s * 1000),
            "decode_speed_tps": round(self.decode_speed, 1),
        }


# ── Engine wrapper ──────────────────────────────────────────────────────

class InferenceEngine:
    """Wraps HomeSeekInferenceEngine with session management and stats."""

    def __init__(self, weight_dir: str = "weights", hot_experts: str = "hot_experts.json",
                 verbose: bool = False):
        self.weight_dir = weight_dir
        self.tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
        self.hot_experts_path = os.path.abspath(hot_experts)
        if not os.path.exists(self.hot_experts_path):
            _log(f"WARNING: hot_experts file not found: {self.hot_experts_path}")
            _log("  Only hash-layer experts will be pinned at startup.")
        else:
            _log(f"Hot experts file: {self.hot_experts_path}")
        self._init_tokenizer()
        self.engine = HomeSeekInferenceEngine(
            weight_dir, device="cuda", verbose=verbose,
            hot_experts_path=self.hot_experts_path,
        )
        self.thinking_mode = False

    def _init_tokenizer(self):
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)

    def set_thinking(self, enabled: bool):
        self.thinking_mode = enabled

    def encode_messages(self, messages: list[dict]) -> torch.Tensor:
        from home_seek.encoding_dsv4 import encode_messages
        mode = "thinking" if self.thinking_mode else "chat"
        text = encode_messages(messages, thinking_mode=mode)
        return self.tokenizer.encode(text, return_tensors="pt").to(self.engine.device)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 0.0, stream_callback=None) -> tuple[list[int], RequestStats]:
        stats = RequestStats()
        stats.encoding_tokens = input_ids.shape[1]

        t0 = time.perf_counter()
        result = self.engine.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, stream_callback=stream_callback,
        )
        torch.cuda.synchronize()
        total = time.perf_counter() - t0

        prefill = result.get('prefill_time_s', 0)
        if prefill > 0:
            stats.encoding_time_s = prefill
            stats.ttft_s = prefill
            stats.decode_time_s = total - prefill
        else:
            stats.encoding_time_s = 0
            stats.ttft_s = 0
            stats.decode_time_s = total

        raw_ids = result['tokens'][0].tolist()
        prompt_len = input_ids.shape[1]
        new_ids = raw_ids[prompt_len:]
        # Filter out stop tokens (token 1)
        generated = [t for t in new_ids if t != 1]
        stats.generated_tokens = len(generated)

        return generated, stats

    def decode_tokens(self, token_ids: list[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)


# ── OpenAI API handler (stateless, uses engine from global) ─────────────

def build_chat_response(engine: InferenceEngine, messages: list[dict],
                        max_tokens: int, temperature: float, stream: bool,
                        request_id: str, model: str):
    """Build chat completion response (non-streaming)."""
    input_ids = engine.encode_messages(messages)

    def _noop(_): pass
    generated, stats = engine.generate(
        input_ids, max_new_tokens=max_tokens,
        temperature=temperature,
        stream_callback=_noop if stream else None,
    )

    content = engine.decode_tokens(generated)

    response = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": stats.encoding_tokens,
            "completion_tokens": stats.generated_tokens,
            "total_tokens": stats.encoding_tokens + stats.generated_tokens,
        },
        "stats": stats.dict(),
    }
    return response, stats


async def build_stream_chunks(engine: InferenceEngine, messages: list[dict],
                               max_tokens: int, temperature: float,
                               request_id: str, model: str) -> AsyncGenerator[str, None]:
    """Stream chat completion as SSE chunks."""
    input_ids = engine.encode_messages(messages)
    collected_ids = []

    def stream_callback(token_id: int):
        collected_ids.append(token_id)

    generated, stats = engine.generate(
        input_ids, max_new_tokens=max_tokens,
        temperature=temperature,
        stream_callback=stream_callback,
    )

    # Yield each token as SSE (buffer IDs to properly assemble multi-byte UTF-8)
    buf = _TokenBuffer(engine.tokenizer)
    for tid in collected_ids:
        if tid == 1:
            continue
        content = buf.add(tid)
        if not content:
            continue
        chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Flush any held-back text (incomplete byte sequences at end of generation)
    rest = buf.flush()
    if rest:
        chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": rest},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Final chunk with usage
    full_content = engine.decode_tokens(generated)
    final = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": stats.encoding_tokens,
            "completion_tokens": stats.generated_tokens,
            "total_tokens": stats.encoding_tokens + stats.generated_tokens,
        },
        "stats": stats.dict(),
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ── FastAPI app ─────────────────────────────────────────────────────────

_engine_instance: InferenceEngine | None = None

def create_app(engine: InferenceEngine):
    """Create FastAPI app with the given engine instance."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="Home-Seek API", version="0.1.0")
    global _engine_instance
    _engine_instance = engine

    class ChatRequest(BaseModel):
        model: str = "home-seek"
        messages: list[dict]
        max_tokens: int = 2048
        temperature: float = 0.0
        stream: bool = False
        thinking: bool | None = None
        reasoning_effort: str | None = None

    class ModelInfo(BaseModel):
        id: str
        object: str = "model"
        created: int = 0
        owned_by: str = "home-seek"

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [ModelInfo(id="home-seek", created=int(time.time())).model_dump()]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        eng = _engine_instance
        if eng is None:
            raise HTTPException(503, "Engine not initialized")

        # Thinking mode: request param > global setting
        thinking = req.thinking if req.thinking is not None else (
            req.reasoning_effort == "high" if req.reasoning_effort else eng.thinking_mode
        )
        old_mode = eng.thinking_mode
        eng.set_thinking(thinking)

        rid = uuid.uuid4().hex[:12]
        try:
            if req.stream:
                return StreamingResponse(
                    build_stream_chunks(eng, req.messages, req.max_tokens,
                                        req.temperature, rid, req.model),
                    media_type="text/event-stream",
                )
            response, _ = build_chat_response(
                eng, req.messages, req.max_tokens, req.temperature, False, rid, req.model)
            return response
        finally:
            eng.set_thinking(old_mode)

    return app


# ── Thread-based HTTP server (avoids uvicorn forking issues with CUDA) ──

def start_http_server(engine: InferenceEngine, host: str = "0.0.0.0", port: int = 8000):
    """Start a thread-based HTTP server. No forking → safe with CUDA."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import traceback

    # Log engine state
    _log("=" * 50)
    _log(f"Server starting on {host}:{port}")
    _log(f"  Hot experts pinned: {len(engine.engine.expert_cache.pinned)}")
    _log(f"  Cache max entries:  {engine.engine.expert_cache.max_experts}")
    _log(f"  Thinking mode:      {engine.thinking_mode}")
    cache_size = len(engine.engine.expert_cache)
    _log(f"  Current cache size: {cache_size} entries")
    _log(f"  Hot experts file:   {engine.hot_experts_path}")
    _log("=" * 50)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, data: dict, status: int = 200):
            msg = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send_json({
                    "object": "list",
                    "data": [{"id": "home-seek", "object": "model", "created": int(time.time()),
                              "owned_by": "home-seek"}],
                })
                _log("200 GET /v1/models")
            else:
                self._send_json({"error": "not found"}, 404)
                _log(f"404 GET {self.path}")

        def do_POST(self):
            t0 = time.time()
            if self.path != "/v1/chat/completions":
                _log(f"404 POST {self.path}")
                return self._send_json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
                body_data = json.loads(self.rfile.read(n).decode())
            except Exception as e:
                _log(f"400 POST bad request: {e}")
                return self._send_json({"error": f"bad request: {e}"}, 400)

            stream = body_data.get("stream", False)
            thinking = body_data.get("thinking")
            if thinking is not None:
                old_mode = engine.thinking_mode
                engine.set_thinking(thinking)
            else:
                old_mode = None

            rid = uuid.uuid4().hex[:12]
            model = body_data.get("model", "home-seek")
            last_msg = (body_data.get("messages", []) or [{}])[-1].get("content", "")[:50]
            _log(f"POST /v1/chat/completions  stream={stream}  msg=\"{last_msg}\"")

            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")  # disable nginx buffering
                    self.end_headers()
                    input_ids = engine.encode_messages(body_data.get("messages", []))
                    gen_done = [False]
                    gen_stats = [None]

                    def _write_chunk(txt: str):
                        if not txt:
                            return
                        c = {"id": f"chatcmpl-{rid}", "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "delta": {"content": txt}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()

                    tok_buf = _TokenBuffer(engine.tokenizer)
                    def cb(tok_id):
                        nonlocal tok_buf
                        if tok_id == 1:
                            tok_buf.reset()
                            return
                        content = tok_buf.add(tok_id)
                        if content:
                            _write_chunk(content)

                    new_ids, req_stats = engine.generate(input_ids,
                        max_new_tokens=body_data.get("max_tokens", 2048),
                        temperature=body_data.get("temperature", 0.0),
                        stream_callback=cb)
                    rest = tok_buf.flush()
                    if rest:
                        _write_chunk(rest)
                    dt = time.time() - t0
                    _log(f"200 stream done in {dt:.1f}s  (gen={len(new_ids)} tok)")
                    stats_dict = {
                        "encoding_tokens": req_stats.encoding_tokens,
                        "encoding_time_ms": round(req_stats.encoding_time_s * 1000),
                        "encoding_speed_tps": round(req_stats.encoding_speed, 1),
                        "generated_tokens": req_stats.generated_tokens,
                        "decode_time_ms": round(req_stats.decode_time_s * 1000),
                        "decode_speed_tps": round(req_stats.decode_speed, 1),
                    }
                    final = {"id": f"chatcmpl-{rid}", "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                             "stats": stats_dict}
                    self.wfile.write(f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    resp, stats = build_chat_response(engine, body_data.get("messages", []),
                        body_data.get("max_tokens", 2048), body_data.get("temperature", 0.0),
                        False, rid, model)
                    dt = time.time() - t0
                    _log(f"200 done in {dt:.1f}s  (gen={stats.generated_tokens} tok)")
                    self._send_json(resp)
            except Exception as e:
                tb = traceback.format_exc()
                _log(f"500 {e}")
                _log(tb)
                try:
                    self._send_json({"error": str(e)}, 500)
                except Exception:
                    pass
            finally:
                if old_mode is not None:
                    engine.set_thinking(old_mode)

        def log_message(self, *args):
            pass

    server = HTTPServer((host, port), Handler)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"\nAPI server on http://{host}:{port}")
    print("  POST /v1/chat/completions  (OpenAI-compatible)")
    print("  GET  /v1/models")
    print("  Thinking mode: send thinking=true in request body")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


# ── Interactive terminal mode ───────────────────────────────────────────

def interactive_mode(engine: InferenceEngine, show_stats: bool = True):
    """REPL loop with commands."""
    print(f"\n{'='*60}")
    print("Home-Seek Interactive")
    print(f"{'='*60}")
    print(f"Commands:  /think     toggle thinking mode (current: {engine.thinking_mode})")
    print("           /stats     show statistics summary")
    print("           /help      this help")
    print("           /quit      exit")
    print(f"{'='*60}\n")

    history = []
    total_stats = RequestStats()
    n_requests = 0

    while True:
        try:
            line = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd = line[1:].lower()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print("  /think     toggle thinking mode")
                print("  /stats     show accumulated stats")
                print("  /help      this help")
                print("  /quit      exit")
            elif cmd == "think":
                engine.set_thinking(not engine.thinking_mode)
                print(f"  Thinking mode: {'ON' if engine.thinking_mode else 'OFF'}")
            elif cmd == "stats":
                if n_requests == 0:
                    print("  No requests yet")
                else:
                    avg_enc = total_stats.encoding_speed / n_requests if n_requests > 0 else 0
                    avg_dec = total_stats.decode_speed / n_requests if n_requests > 0 else 0
                    print(f"  Requests: {n_requests}")
                    print(f"  Avg prefill: {total_stats.encoding_tokens/n_requests:.0f} tok, "
                          f"{total_stats.encoding_time_s/n_requests*1000:.0f}ms")
                    print(f"  Avg encode speed: {avg_enc:.1f} t/s")
                    print(f"  Avg TTFT: {total_stats.ttft_s/n_requests*1000:.0f}ms")
                    print(f"  Avg decode speed: {avg_dec:.1f} t/s")
                    print(f"  Total generated tokens: {total_stats.generated_tokens}")
            else:
                print(f"  Unknown command: {line}")
            continue

        # Chat
        n_requests += 1
        messages = [{"role": "user", "content": line}]
        input_ids = engine.encode_messages(messages)

        t_start = time.perf_counter()

        if show_stats:
            print("  ", end="", flush=True)

        collected = []
        tok_buf = _TokenBuffer(engine.tokenizer)
        def cb(tok_id):
            collected.append(tok_id)
            if show_stats:
                if tok_id == 1:
                    tok_buf.reset()
                    return
                text = tok_buf.add(tok_id)
                if text:
                    print(text, end="", flush=True)

        generated, stats = engine.generate(
            input_ids, max_new_tokens=2048, temperature=0.0,
            stream_callback=cb if show_stats else None,
        )
        if show_stats:
            rest = tok_buf.flush()
            if rest:
                print(rest, end="", flush=True)
        total_s = time.perf_counter() - t_start

        if not show_stats:
            content = engine.decode_tokens(generated)
            print(content)
        else:
            print()

        # Accumulate
        total_stats.encoding_tokens += stats.encoding_tokens
        total_stats.encoding_time_s += stats.encoding_time_s
        total_stats.decode_time_s += stats.decode_time_s
        total_stats.generated_tokens += stats.generated_tokens
        total_stats.ttft_s += stats.ttft_s

        if show_stats:
            print(f"  [{stats.summary()}]  total={total_s*1000:.0f}ms")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Home-Seek API Server")
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--hot-experts", default="hot_experts.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive terminal mode (no HTTP server)")
    parser.add_argument("--no-stats", action="store_true",
                        help="Disable per-response stats in interactive mode")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Initializing engine...")
    t0 = time.time()
    engine = InferenceEngine(
        weight_dir=args.weight_dir,
        hot_experts=args.hot_experts,
        verbose=args.verbose,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"  Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"  Hot experts pinned: {len(engine.engine.expert_cache.pinned)}")

    if args.interactive:
        interactive_mode(engine, show_stats=not args.no_stats)
    else:
        start_http_server(engine, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
