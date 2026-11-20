"""Plug-in surface: attach VENOM to any host.

    from venom.shell import attach
    with attach(model, tokenizer) as host:
        host.chat("...")
"""
from venom.shell.attach import AttachedHost, attach, fingerprint

__all__ = ["attach", "fingerprint", "AttachedHost"]
