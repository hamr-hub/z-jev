# Z-Jev deployment guide

This document covers the four supported ways to put a Z-Jev model behind a
public endpoint: bare metal, Docker, Compose, and Kubernetes / reverse
proxies. It also lists every environment variable Z-Jev understands.

## 1. Bare metal

### Prerequisites

* Python 3.10 or newer (3.11 recommended for parity with the CI image).
* A CPU-only PyTorch install: `pip install -r requirements/cpu.txt`.
* A trained checkpoint at `checkpoints/<NAME>/model.pt` (or any path).

### Install

```bash
git clone https://github.com/hamr-hub/z-jev
cd z-jev
pip install -r requirements/cpu.txt
pip install -e .
```

### Train (optional)

The repo ships a CPU-friendly trainer:

```bash
python -m z_jev.train --out checkpoints/tiny --steps 80 --n-train 128
```

### Run the API

```bash
export ZJEV_CHECKPOINT=/abs/path/to/checkpoints/tiny/model.pt
export ZJEV_API_KEY=please-change-me   # optional, enables Bearer auth
python -m z_jev.serve --host 0.0.0.0 --port 8080
```

Smoke test:

```bash
curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $ZJEV_API_KEY" \
     -X POST http://127.0.0.1:8080/v1/evaluate \
     --data @examples/support_ticket_request.json | jq
```

## 2. Docker (single container)

```bash
docker build -t z-jev:dev -f Dockerfile .
docker run --rm -p 8080:8000 \
    -v "$PWD/checkpoints:/app/checkpoints:ro" \
    -e ZJEV_CHECKPOINT=/app/checkpoints/tiny/model.pt \
    -e ZJEV_API_KEY=please-change-me \
    z-jev:dev
```

The image runs as the unprivileged `zjev` user, listens on port 8000
inside the container, exposes a `HEALTHCHECK` that curls `/healthz`,
and ships CPU-only wheels (~150 MB compressed).

## 3. docker compose

```bash
# .env (or pass inline):
cat > .env <<EOF
ZJEV_API_KEY=please-change-me
ZJEV_CHECKPOINT=/app/checkpoints/tiny/model.pt
EOF

docker compose up -d --build
docker compose logs -f z-jev
```

The compose stack mounts `./checkpoints` (read-only) and `./data`
(read-only) so you can drop new JSONL training files into `./data` and
mount them into the container at the same path on every container.

## 4. Reverse proxy (nginx)

Z-Jev serves `/healthz`, `/readyz`, and `/v1/evaluate`. A typical nginx
front-end:

```nginx
upstream zjev {
    server 127.0.0.1:8080;
    keepalive 16;
}

server {
    listen 80;
    server_name zjev.example.com;

    location / {
        proxy_pass http://zjev;
        proxy_set_header Host $host;
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
```

`X-Request-ID` round-trips through Z-Jev's middleware and is echoed on
every response, so upstream load balancers can correlate.

## 5. systemd unit

```ini
[Unit]
Description=Z-Jev decision API
After=network.target

[Service]
Type=simple
User=zjev
WorkingDirectory=/opt/z-jev
Environment=ZJEV_CHECKPOINT=/opt/z-jev/checkpoints/tiny/model.pt
Environment=ZJEV_API_KEY=please-change-me
ExecStart=/opt/z-jev/.venv/bin/python -m z_jev.serve --host 127.0.0.1 --port 8080
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=z-jev

[Install]
WantedBy=multi-user.target
```

Enable with:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now z-jev
```

## 6. Kubernetes

Use the `/healthz` route for `livenessProbe` (always 200 while the
process is up) and `/readyz` for `readinessProbe` (returns 503 until
the checkpoint is loaded). Example probe:

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 6
```

Mount the checkpoint via a `PersistentVolumeClaim` or a `ConfigMap`-like
artifact (the `.pt` file is just `torch.save` output).

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `ZJEV_CHECKPOINT` | *(unset)* | Path to the model checkpoint to load on startup. Equivalent to passing `--checkpoint`. |
| `ZJEV_API_KEY` | *(unset)* | When set, every request must carry `Authorization: Bearer <key>` except `/healthz` and `/readyz`. |
| `ZJEV_HOST` | `127.0.0.1` | Default bind address for the CLI entry point. |
| `ZJEV_PORT` | `8080` | Default bind port for the CLI entry point. |
| `ZJEV_MAX_REQUEST_BYTES` | `262144` | Maximum request body size in bytes. Larger requests get a 422 `request_too_large`. |
| `ZJEV_MAX_QUESTIONS` | `32` | Maximum number of questions per request. |
| `ZJEV_MAX_OPTIONS_PER_CHOICE` | `16` | Maximum number of options in a Choice primitive. |
| `ZJEV_MAX_LEVELS_PER_SCORE` | `8` | Maximum number of levels in a Score primitive. |
| `ZJEV_MAX_STATE_CHARS` | `8192` | Maximum length of the state text. |
| `ZJEV_LOG_STATE_CHARS` | `40` | How many characters of the state text get written to the JSON-lines access log (avoid leaking PII). |

## Upgrades and rollbacks

Z-Jev ships checkpoints as a single `model.pt` file. The recommended
upgrade flow is:

1. Pull the new image (`docker compose pull` or rebuild).
2. Drain traffic (the readiness probe will go 503 while the new
   checkpoint loads).
3. Start the new container; the old one stays up until you `docker
   compose up -d` the new stack.
4. Verify `/readyz` returns 200 before re-routing traffic.
5. Keep the previous `model.pt` on disk so you can roll back by simply
   pointing `ZJEV_CHECKPOINT` at the old file and restarting the
   container.

## What is **not** covered

* TLS termination -- put Z-Jev behind a reverse proxy (nginx / Traefik /
  Cloudflare) that terminates HTTPS.
* Horizontal scaling -- Z-Jev is currently single-process per replica
  because the tiny model is in-memory; scale by running multiple
  replicas behind a load balancer rather than multi-processing inside a
  single container.
* Authentication beyond a static Bearer token. For OAuth2 / mTLS,
  delegate to the reverse proxy or sidecar (e.g. Envoy + SPIFFE).
