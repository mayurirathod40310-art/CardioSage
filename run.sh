#!/usr/bin/env bash
# Linux/macOS equivalent of setup.ps1 + run.ps1 (same steps, same checks). Usage: ./run.sh [--check] [--retrain] [--port N]
set -euo pipefail
cd "$(dirname "$0")"
PORT=8000; CHECK=0; RETRAIN=0
while [ $# -gt 0 ]; do case "$1" in --check) CHECK=1;; --retrain) RETRAIN=1;; --port) PORT="$2"; shift;; *) echo "unknown option $1"; exit 2;; esac; shift; done
fail() { echo "ERROR: $*" >&2; exit 1; }

command -v python3 >/dev/null || fail "Python 3.10-3.13 not found"
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)' 2>/dev/null || fail "Python 3.10-3.13 required (found $(python3 -V))"
command -v node >/dev/null || fail "Node.js 18+ not found"
[ "$(node -p 'process.versions.node.split(".")[0]')" -ge 18 ] || fail "Node 18+ required"
[ -f data/heart.csv ] || fail "data/heart.csv missing"

[ -x .venv/bin/python ] || { echo "==> Creating .venv"; python3 -m venv .venv; }
H=$(sha256sum backend/requirements.txt | cut -d' ' -f1)
if [ ! -f .venv/.requirements.sha256 ] || [ "$(cat .venv/.requirements.sha256)" != "$H" ]; then
  echo "==> Installing Python dependencies"; .venv/bin/python -m pip install --disable-pip-version-check -q -r backend/requirements.txt && echo "$H" > .venv/.requirements.sha256
else echo "    Python dependencies up to date"; fi

PH=$(sha256sum frontend/package.json | cut -d' ' -f1)
if [ ! -d frontend/node_modules ] || [ ! -f frontend/node_modules/.package-hash ] || [ "$(cat frontend/node_modules/.package-hash)" != "$PH" ]; then
  echo "==> Installing JavaScript dependencies"; (cd frontend && npm install --no-audit --no-fund) && echo "$PH" > frontend/node_modules/.package-hash
else echo "    JavaScript dependencies up to date"; fi
if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src frontend/index.html frontend/package.json -newer frontend/dist/index.html -print -quit)" ]; then
  echo "==> Building frontend"; (cd frontend && npm run build)
else echo "    Frontend build up to date"; fi
if [ "$RETRAIN" = 1 ] || [ ! -f models/current.json ]; then echo "==> Training models"; (cd backend && ../.venv/bin/python -m cardiosage.training); else echo "    Trained model found"; fi

mkdir -p .run storage
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then fail "Port $PORT already in use (use --port)"; fi
echo "==> Starting backend"
(cd backend && exec ../.venv/bin/python -m uvicorn cardiosage.main:app --host 127.0.0.1 --port "$PORT") > .run/backend.out.log 2> .run/backend.err.log &
BE=$!
trap 'kill $BE 2>/dev/null || true; echo; echo Stopped.' EXIT INT TERM
for i in $(seq 1 90); do
  kill -0 $BE 2>/dev/null || { tail -25 .run/backend.err.log; fail "backend exited; see .run/backend.err.log"; }
  curl -sf "http://127.0.0.1:$PORT/api/health" >/tmp/cs_health.json && break; sleep 0.7
done
[ -s /tmp/cs_health.json ] || fail "backend did not become healthy"
if [ "$CHECK" = 1 ]; then
  curl -sf "http://127.0.0.1:$PORT/" | grep -q CardioSage || fail "UI not served"
  P=$(curl -sf -X POST "http://127.0.0.1:$PORT/api/predict" -H 'Content-Type: application/json' -d '{"features":{"age":57,"sex":1,"cp":0,"trestbps":140,"chol":241,"fbs":0,"restecg":1,"thalach":123,"exang":1,"oldpeak":0.2,"slope":1,"ca":0,"thal":3},"save":false}')
  echo "$P" | python3 -c 'import sys,json; j=json.load(sys.stdin); print("    OK: health", json.load(open("/tmp/cs_health.json"))["status"], "| UI served | P(target=1) = %.3f from %s" % (j["probability_class1"], j["algorithm"]))'
  exit 0
fi
echo; echo "  CardioSage is running"; echo "    App:      http://127.0.0.1:$PORT"; echo "    API docs: http://127.0.0.1:$PORT/docs"; echo "  Ctrl+C to stop."
wait $BE
