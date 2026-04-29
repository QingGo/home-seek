"""Entry point for `home-seek` CLI."""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error

from home_seek.api_server import InferenceEngine, start_http_server

MODELSCOPE_REPO = "deepseek-ai/DeepSeek-V4-Flash"
HUGGINGFACE_REPO = "deepseek-ai/DeepSeek-V4-Flash"


# ── ANSI helpers ─────────────────────────────────────────────────────

def _supports_color():
    """Check if output should use ANSI color codes.

    Uses TERM detection rather than isatty(), because uv run may not
    properly report the TTY status even when output goes to a terminal.
    """
    term = os.environ.get("TERM", "")
    if term in ("dumb", "unknown", ""):
        return False
    return True

def _B(code: str) -> str:
    """Return ANSI escape code if color is supported, else empty string."""
    term = os.environ.get("TERM", "")
    if term in ("dumb", "unknown", ""):
        return ""
    return f"\033[{code}m"

BOLD = _B("1")
GREEN = _B("92")
CYAN = _B("96")
YELLOW = _B("93")
DIM = _B("2")
RESET = _B("0")


# ── download ────────────────────────────────────────────────────────────

def cmd_download(args):
    dest = os.path.abspath(args.dir)
    if os.path.exists(dest):
        print(f"Error: directory already exists: {dest}")
        print("  Remove it first, or choose a different path with --dir")
        sys.exit(1)

    source = args.source
    repo = MODELSCOPE_REPO if source == "modelscope" else HUGGINGFACE_REPO
    service = "ModelScope" if source == "modelscope" else "Hugging Face"

    print(f"Downloading {repo} from {service} ...")
    print(f"  Target: {dest}")
    print("  Size:   ~150 GB (46 safetensor files)")
    print("  This may take a long time depending on your connection.")
    print()

    try:
        if source == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download(repo, cache_dir=dest)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=dest, local_dir_use_symlinks=False)
        print(f"\nDone! Weights saved to {dest}")
    except ImportError as e:
        print(f"Error: missing dependency: {e}")
        print("  For ModelScope: pip install modelscope")
        print("  For HuggingFace: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)


# ── server ──────────────────────────────────────────────────────────────

def cmd_server(args):
    weight_dir = os.path.abspath(args.weight_dir)
    config_path = os.path.join(weight_dir, "config.json")

    if not os.path.isdir(weight_dir):
        print(f"Error: weight directory not found: {weight_dir}")
        _print_download_help()
        sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"Error: {config_path} not found")
        print("  The directory exists but doesn't contain model weights.")
        _print_download_help()
        sys.exit(1)

    import torch
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    print("Home-Seek Server", flush=True)
    print(f"  Model: {weight_dir}", flush=True)
    print("  Initializing...", end=" ", flush=True)
    t0 = time.time()

    hot_path = os.path.join(weight_dir, "..", "hot_experts.json")
    engine = InferenceEngine(
        weight_dir=weight_dir,
        hot_experts=hot_path,
        verbose=args.verbose,
    )
    dt = time.time() - t0
    print(f"done in {dt:.1f}s", flush=True)
    print(f"  GPU: {torch.cuda.get_device_properties(0).name}", flush=True)
    print(f"  Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    print(f"  Hot experts pinned: {len(engine.engine.expert_cache.pinned)}", flush=True)

    start_http_server(engine, host=args.host, port=args.port)


def _print_download_help():
    print()
    print("  To download model weights:")
    print("    home-seek download                     # ModelScope (China, fast)")
    print("    home-seek download --source huggingface # Hugging Face")
    print("    home-seek download --dir /path/to/weights")
    print()


# ── cli ─────────────────────────────────────────────────────────────────

def cmd_cli(args):
    host = args.host
    port = args.port
    base = f"http://{host}:{port}"

    # Verify server
    try:
        urllib.request.urlopen(f"{base}/v1/models", timeout=5)
    except urllib.error.URLError as e:
        print(f"Error: cannot connect to Home-Seek server at {base}")
        if isinstance(e.reason, ConnectionRefusedError):
            print("  The server is not running.")
        else:
            print(f"  {e.reason}")
        print()
        print("  Start the server first:")
        print(f"    home-seek server --port {port}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: cannot reach server at {base}")
        print(f"  {e}")
        sys.exit(1)

    thinking = False

    def stream_chat(text):
        """Send chat via SSE streaming, yield (token_text, final_stats) pairs."""
        body = json.dumps({
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 2048, "temperature": 0.0,
            "stream": True,
            "thinking": thinking or None,
        }).encode()
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=600)
        stats = {}
        while True:
            line = resp.readline().decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    for c in data.get("choices", []):
                        delta = c.get("delta", {})
                        txt = delta.get("content", "")
                        if txt:
                            yield txt, None
                    if data.get("stats"):
                        stats = data["stats"]
                except json.JSONDecodeError:
                    pass
            elif not line:
                break
        yield "", stats

    # readline + ANSI prompt: wrap escape sequences in \001/\002
    # so readline doesn't count them as visible characters (prevents
    # cursor corruption & backspace eating the prompt).
    try:
        import readline
    except ImportError:
        readline = None

    import logging
    _lg = logging.getLogger("home-seek")
    _lg.info(f"ANSI: TERM={os.environ.get('TERM','')!r} "
             f"GREEN={GREEN.encode('utf-8').hex()!r} "
             f"RESET={RESET.encode('utf-8').hex()!r} "
             f"GREEN_len={len(GREEN)}")

    print()
    banner = f"{'='*60}"
    print(f"{banner}")
    print(f"  {GREEN}Home-Seek Chat{RESET}  (server: {base})")
    print(f"{banner}")
    print(f"  {DIM}/think{RESET}    toggle thinking mode")
    print(f"  {DIM}/stats{RESET}    show last exchange stats")
    print(f"  {DIM}/help{RESET}     this help")
    print(f"  {DIM}/quit{RESET}     exit")
    print(f"{banner}")
    print()

    # Ensure stdout can handle UTF-8 (emojis, CJK, etc.)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Wrap ANSI codes in \001/\002 for readline cursor accounting.
    # Without readline, use plain ANSI codes (no \001/\002 needed).
    _PROMPT = (f"\001{GREEN}\002>>>\001{RESET}\002 " if readline
               else f"{GREEN}>>>{RESET} ")

    last_s = None
    while True:
        try:
            raw = input(_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        if raw.startswith("/"):
            cmd = raw[1:].lower()
            if cmd in ("quit", "exit", "q"):
                print("bye")
                break
            elif cmd == "help":
                print(f"  {DIM}/think{RESET}     toggle thinking mode")
                print(f"  {DIM}/stats{RESET}     show stats for last exchange")
                print(f"  {DIM}/help{RESET}      this help")
                print(f"  {DIM}/quit{RESET}      exit")
            elif cmd == "think":
                thinking = not thinking
                print(f"  Thinking mode: {BOLD}{'ON' if thinking else 'OFF'}{RESET}")
            elif cmd == "stats":
                if not last_s:
                    print("  No requests yet")
                else:
                    s = last_s
                    print(f"  {CYAN}Last exchange:{RESET}")
                    print(f"    Prefill: {s['encoding_tokens']}tok/{s['encoding_time_ms']}ms"
                          f" ({s['encoding_speed_tps']}t/s)")
                    print(f"    Decode:  {s['generated_tokens']}tok/{s['decode_time_ms']}ms"
                          f" ({s['decode_speed_tps']}t/s)")
            else:
                print(f"  Unknown: {raw}")
            continue

        # Send + stream
        t_start = time.time()
        print(f"  {DIM}", end="", flush=True)
        try:
            collected = []
            for token_text, stats in stream_chat(raw):
                if token_text:
                    sys.stdout.buffer.write(token_text.encode('utf-8'))
                    sys.stdout.buffer.flush()
                    collected.append(token_text)
                if stats:
                    last_s = stats
            elapsed = time.time() - t_start
        except urllib.error.HTTPError as e:
            detail = e.read().decode()
            print(f"\n  {YELLOW}Server error ({e.code}):{RESET}")
            try:
                msg = json.loads(detail).get("error", detail)
                print(f"  {msg}")
            except json.JSONDecodeError:
                print(f"  {detail}")
            continue
        except Exception as e:
            print(f"\n  {YELLOW}Error:{RESET} {e}")
            continue

        print()

        if last_s:
            s = last_s
            parts = []
            if s.get('encoding_time_ms'):
                parts.append(f"prefill {s['encoding_tokens']}tok/{s['encoding_time_ms']}ms"
                             f"({s['encoding_speed_tps']}t/s)")
            if s.get('decode_time_ms'):
                parts.append(f"decode {s['generated_tokens']}tok/{s['decode_time_ms']}ms"
                             f"({s['decode_speed_tps']}t/s)")
            parts.append(f"total {elapsed*1000:.0f}ms")
            print(f"  {DIM}[{'  '.join(parts)}]{RESET}")


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Home-Seek CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download model weights")
    p.add_argument("--dir", default="weights",
                   help="Target directory (default: weights/)")
    p.add_argument("--source", choices=["modelscope", "huggingface"],
                   default="modelscope",
                   help="Download source (default: modelscope)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("server", help="Start API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--weight-dir", default="weights",
                   help="Model weights directory (default: weights/)")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_server)

    p = sub.add_parser("cli", help="Interactive chat client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
