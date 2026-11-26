#!/usr/bin/env python3
"""Serve the experiment atlas and its repository evidence with Python's stdlib."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        url_path = unquote(urlsplit(path).path)
        if url_path.startswith('/experiment-artifacts/'):
            return str(ROOT.parent/'experiment_results'/url_path.removeprefix('/experiment-artifacts/'))
        return super().translate_path(path)

    def do_GET(self):
        if urlsplit(self.path).path == '/':
            self.send_response(302)
            self.send_header('Location', '/docs/experiment_atlas/')
            self.end_headers()
            return
        super().do_GET()

    def send_head(self):
        # Serve the atlas and evidence, without exposing Git metadata or unrelated checkout files.
        path = Path(self.translate_path(self.path)).resolve()
        allowed = [ROOT / 'docs', ROOT / 'results', ROOT / 'experiments/libero/specs']
        if urlsplit(self.path).path.startswith('/experiment-artifacts/'):
            base = (ROOT.parent/'experiment_results').resolve()
            if (not path.is_relative_to(base) or 'eval_artifacts' not in path.relative_to(base).parts
                    or path.suffix not in ('.png','.mp4','.npz','.json') or not path.is_file()):
                self.send_error(404)
                return None
            return super().send_head()
        if not any(path.is_relative_to(root.resolve()) for root in allowed) or any(part.startswith('.') for part in Path(unquote(urlsplit(self.path).path)).parts):
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(ROOT)))
    print(f'Experiment Atlas: http://localhost:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
