"""The `venom` command line interface.

    venom serve                      # MCP stdio server (tools for any MCP client)
    venom fingerprint --model gpt2   # describe a host without attaching
    venom attach --model gpt2       # attach, generate, detach - proves it works
    venom proxy --upstream URL ...   # put VENOM in front of ANY API model
    venom remember "key" "value"     # seed a memory file
    venom recall "key" --memory f   # read it back

Everything is zero-dependency except the commands that actually load a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from venom.core.runtime import VenomConfig, VenomRuntime


def _load_model(name: str, dtype: str = "auto"):
    """Load a local model + tokenizer. Requires torch + transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("this command needs torch + transformers: pip install -e '.[torch]'",
              file=sys.stderr)
        raise SystemExit(2)

    kwargs: Dict[str, Any] = {}
    if dtype != "auto":
        kwargs["dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# ---------------------------------------------------------------- commands
def cmd_serve(args) -> int:
    """Run the MCP stdio server."""
    from venom.server import main as server_main

    runtime = VenomRuntime(VenomConfig(session_id=args.session))
    from venom.tools.handlers import ToolHandlers

    # bind the server to our runtime so tools use its store and budget
    import venom.server as server_mod

    def make_server(store=None):
        return ToolHandlers(runtime.store, runtime.budget)

    server_mod.make_server = make_server
    server_main()
    return 0


def cmd_fingerprint(args) -> int:
    from venom.shell.attach import fingerprint

    model, _ = _load_model(args.model, args.dtype)
    print(json.dumps(fingerprint(model, args.model), indent=2))
    return 0


def cmd_attach(args) -> int:
    """Attach, run one prompt, detach, and prove the host is bit-exact."""
    import torch

    from venom.shell.attach import attach

    model, tok = _load_model(args.model, args.dtype)
    ids = tok(args.prompt, return_tensors="pt")

    with torch.no_grad():
        before = model(**ids).logits
    from venom.core.bus import logits_hash
    hash_before = logits_hash(before)

    with attach(model, tok, name_hint=args.model) as host:
        out = host.generate(args.prompt, max_new_tokens=args.max_new_tokens,
                            do_sample=False)
        stats = host.stats()

    with torch.no_grad():
        after = model(**ids).logits

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "output": out,
        "bit_exact_after_detach": bool(torch.equal(before, after)),
        "hash_before": hash_before[:16],
        "hash_after": logits_hash(after)[:16],
        "fingerprint": stats["fingerprint"],
        "telemetry": {k: v for k, v in stats.items() if k != "fingerprint"},
    }
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["bit_exact_after_detach"] else 1


def cmd_proxy(args) -> int:
    from venom.shell.proxy import VenomProxy

    runtime = VenomRuntime(VenomConfig(session_id=args.session))
    proxy = VenomProxy(
        runtime,
        upstream_base_url=args.upstream,
        upstream_api_key=args.api_key or os.environ.get("VENOM_UPSTREAM_KEY"),
        upstream_model=args.upstream_model,
        host=args.host,
        port=args.port,
    )
    proxy.start()
    print(f"VENOM proxy listening on {proxy.base_url}")
    print(f"  upstream: {args.upstream}")
    print("  point your client at the proxy base_url above; ctrl-c to stop")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping")
        proxy.stop()
    return 0


def cmd_remember(args) -> int:
    runtime = VenomRuntime(VenomConfig(session_id=args.session,
                                        persist_path=args.persist))
    out = runtime.remember(args.key, args.value)
    if args.persist:
        out["persistent"] = True
        out["path"] = args.persist
    if args.memory:
        with open(args.memory, "w", encoding="utf-8") as fh:
            json.dump({args.key: args.value}, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


def cmd_recall(args) -> int:
    runtime = VenomRuntime(VenomConfig(session_id=args.session,
                                        persist_path=args.persist))
    if args.memory and os.path.exists(args.memory):
        with open(args.memory, encoding="utf-8") as fh:
            for k, v in json.load(fh).items():
                runtime.remember(k, v)
    print(json.dumps(runtime.recall(args.key), indent=2))
    return 0


def cmd_tools(args) -> int:
    from venom.tools.schema import ALL_TOOLS, TIER2_TOOLS
    tools = TIER2_TOOLS if args.tier == 2 else ALL_TOOLS
    if args.names:
        print(json.dumps({"tools": [t["name"] for t in tools]}, indent=2))
    else:
        print(json.dumps({"tools": tools}, indent=2))
    return 0


def cmd_probe(args) -> int:
    """Run one forward pass and report what the self-read tools see."""
    import torch

    from venom.shell.attach import attach

    model, tok = _load_model(args.model, args.dtype)
    runtime = VenomRuntime(VenomConfig(session_id=args.session, max_calls=20))
    runtime.remember(*(args.remember.split("=", 1) if "=" in args.remember
                        else ("capital of france", "Paris")))

    out = {}
    with attach(model, tok, runtime=runtime, name_hint=args.model) as host:
        ids = tok(args.prompt, return_tensors="pt")
        with torch.no_grad():
            model(**ids)
        for tool, tool_args in [
            ("venom.uncertainty", {}),
            ("venom.check_myself", {}),
            ("venom.why", {"token": args.token}),
            ("venom.where", {"concept": args.concept}),
            ("venom.audit", {}),
        ]:
            try:
                out[tool] = host.call_tool(tool, tool_args)
            except Exception as exc:  # noqa: BLE001
                out[tool] = {"ok": False, "error": str(exc)}

    def trim(obj):
        if isinstance(obj, dict):
            return {k: trim(v) for k, v in obj.items()
                    if k not in ("caveat", "warning", "note", "per_layer", "layers")}
        if isinstance(obj, list):
            return obj[:8]
        return obj

    print(json.dumps({"model": args.model, "prompt": args.prompt,
                      "tools": trim(out)}, indent=2, default=str))
    return 0


def cmd_chat(args) -> int:
    """Interactive chat with VENOM attached."""
    from venom.shell.attach import attach

    model, tok = _load_model(args.model, args.dtype)
    runtime = VenomRuntime(VenomConfig(session_id=args.session))
    for fact in args.remember or []:
        if "=" in fact:
            k, v = fact.split("=", 1)
            runtime.remember(k, v)

    print(f"VENOM attached to {args.model}. Type 'exit' to quit.")
    with attach(model, tok, runtime=runtime, name_hint=args.model) as host:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("exit", "quit"):
                break
            if line.startswith("/"):
                cmd = line[1:].split()
                if cmd and cmd[0] in ("recall", "report", "tools"):
                    print(json.dumps(host.call_tool("venom." + cmd[0],
                                                   {"query": cmd[1]} if len(cmd) > 1 else {}),
                                     indent=2, default=str))
                continue
            out = host.generate(line, max_new_tokens=args.max_new_tokens, do_sample=False)
            print(out)


def cmd_dial(args) -> int:
    """Apply a bounded self-edit and report the drift."""
    import torch

    from venom.core.dials import logit_drift
    from venom.shell.attach import attach

    model, tok = _load_model(args.model, args.dtype)
    runtime = VenomRuntime(VenomConfig(session_id=args.session, enable_dials=True))
    ids = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        before = model(**ids).logits

    with attach(model, tok, runtime=runtime, name_hint=args.model) as host:
        # the dial verifier compares against a baseline; without one it will
        # refuse the edit (and say so) rather than pretend it is safe
        host.capture_baseline(ids)
        out = host.call_tool("venom.dial", {"layer": args.layer, "delta": args.delta})
        with torch.no_grad():
            after = model(**ids).logits
        drift = logit_drift(before, after)
        revert = host.call_tool("venom.dial", {"layer": args.layer, "delta": args.delta,
                                               "apply": False})
        with torch.no_grad():
            restored = model(**ids).logits

    print(json.dumps({
        "model": args.model, "layer": args.layer, "delta": args.delta,
        "dial_result": out,
        "drift": drift,
        "revert_result": revert,
        "bit_exact_after_revert": bool(torch.equal(before, restored)),
    }, indent=2, default=str))
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="venom", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="store_true", help="print the version and exit")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="run the MCP stdio server")
    s.add_argument("--session", default="default")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("fingerprint", help="describe a host model")
    s.add_argument("--model", required=True)
    s.add_argument("--dtype", default="auto")
    s.set_defaults(func=cmd_fingerprint)

    s = sub.add_parser("attach", help="attach, generate, detach, verify bit-exactness")
    s.add_argument("--model", required=True)
    s.add_argument("--prompt", default="The capital of France is")
    s.add_argument("--max-new-tokens", type=int, default=32)
    s.add_argument("--dtype", default="auto")
    s.set_defaults(func=cmd_attach)

    s = sub.add_parser("proxy", help="proxy any OpenAI-compatible upstream")
    s.add_argument("--upstream", required=True, help="e.g. https://api.openai.com/v1")
    s.add_argument("--upstream-model", default=None)
    s.add_argument("--api-key", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--session", default="default")
    s.set_defaults(func=cmd_proxy)

    s = sub.add_parser("remember", help="store a fact")
    s.add_argument("key")
    s.add_argument("value")
    s.add_argument("--memory", default=None, help="write the fact to this json file")
    s.add_argument("--persist", default=None,
                   help="persistent store path (facts survive a restart)")
    s.add_argument("--session", default="default")
    s.set_defaults(func=cmd_remember)

    s = sub.add_parser("recall", help="read a fact back")
    s.add_argument("key")
    s.add_argument("--memory", default=None, help="read facts from this json file")
    s.add_argument("--persist", default=None, help="persistent store path to read")
    s.add_argument("--session", default="default")
    s.set_defaults(func=cmd_recall)

    s = sub.add_parser("tools", help="print the MCP tool schemas")
    s.add_argument("--tier", type=int, choices=[1, 2], default=1)
    s.add_argument("--names", action="store_true", help="print names only")
    s.set_defaults(func=cmd_tools)

    s = sub.add_parser("probe", help="run one pass and report the self-read tools")
    s.add_argument("--model", required=True)
    s.add_argument("--prompt", default="The capital of France is")
    s.add_argument("--token", default=" Paris")
    s.add_argument("--concept", default=" France")
    s.add_argument("--remember", default="capital of france=Paris")
    s.add_argument("--session", default="default")
    s.add_argument("--dtype", default="auto")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("chat", help="interactive chat with VENOM attached")
    s.add_argument("--model", required=True)
    s.add_argument("--remember", action="append", help="key=value facts to seed")
    s.add_argument("--max-new-tokens", type=int, default=48)
    s.add_argument("--session", default="default")
    s.add_argument("--dtype", default="auto")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("dial", help="apply a bounded self-edit and measure the drift")
    s.add_argument("--model", required=True)
    s.add_argument("--layer", type=int, default=0)
    s.add_argument("--delta", type=float, default=0.05)
    s.add_argument("--prompt", default="The capital of France is")
    s.add_argument("--session", default="default")
    s.add_argument("--dtype", default="auto")
    s.set_defaults(func=cmd_dial)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from venom import __version__
        print(__version__)
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
