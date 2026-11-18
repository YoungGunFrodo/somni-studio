"""python -m somni_studio [--config studio.json] [--host H] [--port P] [--open]"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser

from .settings import SettingsError, load_settings, settings_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="somni_studio", description="Somni Studio: brain + memory UI")
    parser.add_argument("--config", help="settings file (default: studio.json next to the package)")
    parser.add_argument("--host", help="override server.host")
    parser.add_argument("--port", type=int, help="override server.port")
    parser.add_argument("--open", action="store_true", help="open the UI in the browser")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    path = settings_path(args.config)
    try:
        settings = load_settings(path)
        overrides = {k: v for k, v in (("host", args.host), ("port", args.port)) if v is not None}
        if overrides:
            settings = settings.merged({"server": overrides})
    except SettingsError as exc:
        print(f"settings error ({path}): {exc}", file=sys.stderr)
        return 2
    if not settings.server.is_loopback:
        logging.warning("listening on %s: anyone who can reach this address can use the studio",
                        settings.server.host)

    import uvicorn

    from .api.app import create_app

    app = create_app(settings, path)
    url = f"http://{'localhost' if settings.server.is_loopback else settings.server.host}:{settings.server.port}/"
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"Somni Studio on {url}  (settings: {path})")
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, log_level=args.log_level,
                ws="websockets-sansio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
