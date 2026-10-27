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

### Protected live model upload

Open FROSS chat, go to **Settings → Upload FROSS model**, then choose the `.pt` checkpoint (up to 90 MiB) and enter the upload password. The page sends the file through the Cloudflare Worker to Hostless. Configure `FROSS_MODEL_UPLOAD_PASSWORD` as a Hostless service secret. The Worker uses the existing `FROSS_GITHUB_TOKEN` secret, forwards it only server-to-server alongside the existing `FROSS_BRAIN_SERVICE_TOKEN`, and never exposes it to browser JavaScript. The existing `FROSS_BRAIN_SERVICE_TOKEN` must be configured in both Cloudflare and Hostless as it already is for chat requests.

Optional Hostless settings:

- `FROSS_MODEL_GITHUB_REPOSITORY`: defaults to `boberbuset132013-web/FROSS`.
- `FROSS_MODEL_MIN_STEPS`: minimum accepted training steps; the server enforces at least 3,000.
- `FROSS_MODEL_VOCAB_SIZE`: required vocabulary size; defaults to 5,000.

The standalone `/admin/model-upload` page is also retained for server-side diagnostics, but the normal workflow is the FROSS chat Settings panel. The upload password is verified on Hostless and rate-limited after repeated failures.

The endpoint checks checkpoint format/version, training steps, vocabulary and context size, exports an ONNX candidate, and smoke-tests the C++ runtime before publishing it. The binary is uploaded as a GitHub Release Asset named `fross-transformer.pt`; `model-storage/current.json` stores its release URL, SHA-256, size and model metadata. This keeps the repository directory small rather than adding a 63-MB binary to every Git commit. Once published, the service atomically switches its in-memory model without rebuilding Docker. The same Release Asset is selected on subsequent deployments, so it does not have to be uploaded again.

**Security note:** keep both secrets exclusively in Hostless's environment/secrets UI. Never put them into the website JavaScript or GitHub source code. If the repository is public, the Release Asset itself is public too; never upload a checkpoint that must remain private to a public repository.

The selected source is reported as `modelSource` in `/health` and `/api/health`.

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
