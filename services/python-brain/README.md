# FROSS hybrid Python/C++ brain

## Architecture

- Python/PyTorch trains the repository's own Transformer from random weights.
- The exact trained checkpoint is exported to a fixed-shape ONNX graph.
- C++ ONNX Runtime executes inference; Python retains prompt formatting, sampling, HTTP and memory/search integration.
- Basic greetings, thanks and goodbyes are handled deterministically so they cannot turn into gibberish. Questions and tasks still use the trained model.
- No pretrained LLM or hosted LLM API is used.

## Container build settings

The Docker build context must be the **repository root** because the Dockerfile copies `fross_transformer.py` and `data/fross_corpus.jsonl` from the root.

- Dockerfile path: `services/python-brain/Dockerfile`
- Build context: repository root (`.`)
- Port: `10000`
- Keep `FROSS_BRAIN_SERVICE_TOKEN` configured as a secret in the hosting platform. Never put the value in this repository.

The image trains the model during the build, exports `/app/fross-transformer.onnx`, compiles `/app/libfross_cpp.so`, and enables the C++ backend.

## Verify deployment

Open `/health` on the Python brain service. A successful hybrid deployment should report:

- `version: "1.3"`
- `generationModel: "FROSS Transformer 1.3"`
- `inferenceBackend: "cpp-onnx"`
- `cppRuntimeLoaded: true`
- `generationModelStatus.loaded: true`

If `inferenceBackend` is `python-torch`, inspect `cppRuntimeStatus` and hosting build logs. The service falls back to Python if the native library cannot be loaded so the health route remains available.

## Local development and automated checks

Install development/test dependencies with `python -m pip install -r requirements.txt`. Production uses the CPU-specific PyTorch wheel and pinned native runtime in the Dockerfile instead.

The GitHub Actions workflow `.github/workflows/fross-hybrid-brain.yml` trains a small temporary model, verifies checkpoint loading, checks prompt-context preservation, exports and executes ONNX, compiles C++, and compares C++ logits with Python logits from the same weights.
