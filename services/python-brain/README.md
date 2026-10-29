# FROSS hybrid Python/C++ brain

## Architecture

- The Transformer is trained offline, outside Hostless deployment builds.
- Hostless only downloads an already-trained checkpoint, validates it, exports an ONNX graph and smoke-tests native inference.
- C++ ONNX Runtime executes inference; Python retains prompt formatting, sampling, HTTP and memory/search integration.
- Basic greetings, thanks and goodbyes are handled deterministically so they cannot turn into gibberish. Questions and tasks still use the trained model.
- No pretrained LLM or hosted LLM API is used.

## Container build settings

The Docker build context must be the **repository root**.

- Dockerfile path: `services/python-brain/Dockerfile`
- Build context: repository root (`.`)
- Port: `10000`
- Keep `FROSS_BRAIN_SERVICE_TOKEN` configured as a secret in the hosting platform. Never put the value in this repository.

The Dockerfile has no training command. Normal Hostless deployments use the default `FROSS_BUILD_BUNDLE_MODEL=0`, so the Docker build does **not** download the checkpoint from Yandex Disk or export ONNX. The service starts with the upload endpoint available even before a model has been uploaded. On startup it looks for the latest published Release Asset `fross-transformer.pt` in `FROSS_MODEL_GITHUB_REPOSITORY` (default: `boberbuset132013-web/FROSS`); if no compatible release exists yet, it stays online with `generationModelLoaded: false` until the model is uploaded from Settings. Once a valid Release Asset exists, future service restarts retrieve and validate it, then export ONNX. CI sets `FROSS_BUILD_BUNDLE_MODEL=1` so the production inference smoke test still exercises a bundled, validated checkpoint.

### Authenticated live model upload

Open FROSS chat, sign in, go to **Settings → Upload FROSS model**, and choose the `.pt` checkpoint (up to 90 MiB). No separate upload password is needed. The Worker verifies the Firebase session, then Hostless independently requires the shared `FROSS_BRAIN_SERVICE_TOKEN` before accepting the upload. The Worker uses the existing `FROSS_GITHUB_TOKEN` secret and forwards it only server-to-server; it is never exposed to browser JavaScript. Keep the same `FROSS_BRAIN_SERVICE_TOKEN` configured in Cloudflare and Hostless.
Optional Hostless settings:

- `FROSS_MODEL_GITHUB_REPOSITORY`: defaults to `boberbuset132013-web/FROSS`.
- `FROSS_MODEL_MIN_STEPS`: minimum accepted training steps; the server enforces at least 3,000.
- `FROSS_MODEL_VOCAB_SIZE`: required vocabulary size; defaults to 5,000.

The standalone `/admin/model-upload` route is informational only; file uploads are accepted exclusively from the authenticated Worker.

The endpoint checks checkpoint format/version, training steps, vocabulary and context size, exports an ONNX candidate, and smoke-tests the C++ runtime before publishing it. The binary is uploaded as a GitHub Release Asset named `fross-transformer.pt`; `model-storage/current.json` stores its release URL, SHA-256, size and model metadata. This keeps the repository directory small rather than adding a 63-MB binary to every Git commit. Once published, the service atomically switches its in-memory model without rebuilding Docker. The same Release Asset is selected on subsequent deployments, so it does not have to be uploaded again.

**Security note:** keep `FROSS_BRAIN_SERVICE_TOKEN` and `FROSS_GITHUB_TOKEN` in server-side secrets only. Never put them into website JavaScript or GitHub source code. If the repository is public, the Release Asset itself is public too; never upload a checkpoint that must remain private to a public repository.

The selected source is reported as `modelSource` in `/health` and `/api/health`.

## HTTP routes and browser checks

The service base URL configured in FROSS may end in `/chat`. Opening that exact URL in a browser sends **GET /chat**; it now returns JSON health/route diagnostics (HTTP 200) instead of a confusing 404. Actual inference still uses **POST /chat** with a JSON body and the shared `Authorization: Bearer FROSS_BRAIN_SERVICE_TOKEN` header. `GET /health`, `GET /`, and `HEAD /health` are health probes. Query strings and a trailing slash are normalized for route matching, so `/chat/?probe=1` does not produce a false 404.

If a deployed service still returns `{"error":"Not found"}` for `GET /chat`, it is likely running an older image; trigger a Hostless rebuild/redeploy from the latest repository commit with Dockerfile `services/python-brain/Dockerfile` and the repository root as build context.

## Verify deployment

Open `/health` on the Python brain service to view model status. The authenticated `POST /ready` endpoint allows the Worker to verify the shared service token as well as model readiness. A successful hybrid deployment should report:

- `version: "1.4"`
- `generationModel: "FROSS Transformer 1.4"`
- `inferenceBackend: "cpp-onnx"`
- `cppRuntimeLoaded: true`
- `generationModelStatus.loaded: true`

If `inferenceBackend` is `python-torch`, inspect `cppRuntimeStatus` and hosting build logs. The service falls back to Python if the native library cannot be loaded so the health route remains available.

## Local development and automated checks

Install development/test dependencies with `python -m pip install -r requirements.txt`. Production uses the CPU-specific PyTorch wheel and pinned native runtime in the Dockerfile instead.

Train and export checkpoints outside Hostless builds. The GitHub Actions workflow `.github/workflows/fross-hybrid-brain.yml` trains a small temporary model for tests, validates the hosted deployment checkpoint, builds the production Docker image without training commands, exercises the HTTP endpoints, exports and executes ONNX, compiles C++, and compares C++ logits with Python logits from identical weights.
