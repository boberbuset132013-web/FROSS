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

The Dockerfile has no training command and no fallback-training path. Every deployment downloads the pre-trained checkpoint from `FROSS_MODEL_URL` (default: the public Yandex Disk checkpoint), verifies that it is a compatible FROSS Transformer 1.4 checkpoint with a 5,000-token vocabulary, block size 320 and at least `FROSS_REQUIRED_MODEL_STEPS` steps, then exports and validates ONNX. Download, compatibility validation, export or native smoke-test failures stop the build. Training must be performed separately, then the new checkpoint must be uploaded to the configured model URL before deploying it.

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
