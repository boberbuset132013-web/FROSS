# FROSS Python Brain 1.6

## Brain 1.6 architecture

- Decoder-only Transformer, implemented in this repository; no hosted LLM API or third-party pretrained LLM is used.
- Target profile: **64,819,040 parameters**, **8,000-token SentencePiece Unigram vocabulary**, **4,096-token context**.
- Architecture: 18 decoder layers, 8 attention heads, embedding width 520, tied token embeddings.
- Training target: **100,000 cumulative optimizer steps**, exposed as 25k → 50k → 75k → 100k stages. A checkpoint cannot be downloaded for website activation until it reaches 100,000 steps.
- Brain 1.5 remains a supported fallback profile (30K vocabulary, 500-token context, approximately 8.97M parameters). Its hidden width and attention heads are incompatible with Brain 1.6, so those weights are not copied into the new model.

## Long-context and memory handling

The attention implementation processes query positions in small blocks and uses PyTorch activation/gradient checkpointing during training. This reduces peak attention/activation memory at the cost of additional computation. Training uses batch size 1. These optimizations do not eliminate the memory needed for model weights, gradients, optimizer state, temporary checkpoint files or the Python runtime. A low-memory 1-GB container is not a suitable training host for this profile.

The normal Docker image installs the CPU PyTorch wheel. CUDA is used only if the installed PyTorch build and host actually expose a compatible GPU; setting device selection to automatic does not turn a CPU-only container into a GPU build. Plan training capacity and persistent disk storage before starting the 100K cycle.

## Training and checkpoint lifecycle

- Website training combines `data/fross_corpus.jsonl` and `data/fross_teacher_corpus.jsonl`, then builds a fresh 8K Unigram tokenizer.
- First stage starts from a new initialization unless a compatible Brain 1.6 checkpoint already exists. The previous Brain 1.5 checkpoint is preserved for rollback.
- Continuation stages resume the saved model, AdamW state and random-number-generator state. The corpus digest and schedule are checked so different data is not silently mixed into the same run.
- The full training checkpoint is retained separately for continuing training.
- At 100K, the server creates a row-wise INT8 inference package with FP16 small vectors. The service upload path accepts that compact package, dequantizes it for loading and validates its actual parameter count and metadata.
- The website upload limit is 90 MiB. The full optimizer checkpoint is not the upload file.

## Container build

The Docker build context must be the repository root.

- Dockerfile: `services/python-brain/Dockerfile`
- Build context: repository root (`.`)
- Port: `10000`
- Required secret: `FROSS_BRAIN_SERVICE_TOKEN`
- Model release repository: `FROSS_MODEL_GITHUB_REPOSITORY` (defaults to `boberbuset132013-web/FROSS`)

Normal builds use `FROSS_BUILD_BUNDLE_MODEL=0`; they do not train a model or bundle the checkpoint. At startup the service checks the latest GitHub Release asset `fross-transformer.pt`. It keeps the existing Brain 1.5 checkpoint until a Brain 1.6 checkpoint has been downloaded, checksum-verified and validated. If no new release is available, the older compatible model remains available as a fallback.

For controlled CI smoke tests only, `FROSS_MODEL_TEST_MODE=1` allows test vocabulary/context/architecture values to be supplied by environment variables. Do not enable test mode in production.

## Authenticated model upload

Open FROSS → Settings → Upload FROSS model and select the compact Brain 1.6 INT8 file only after all 100,000 training steps have completed. The Worker verifies the Firebase session and forwards the request server-to-server; Hostless independently checks `FROSS_BRAIN_SERVICE_TOKEN`. Keep both the Firebase signing configuration and service token out of client code and repository files.

The server validates:

- Model version and format version.
- At least 100,000 training steps.
- Exactly 8,000 tokenizer entries.
- Exactly 4,096 context positions.
- Architecture: 18 layers, 8 heads, width 520.
- Actual parameter count between 64,800,000 and 65,000,000.
- INT8 package integrity and state-dict compatibility.

After validation, the checkpoint can be published as a GitHub Release Asset named `fross-transformer.pt`; the next service restart downloads the latest verified asset. The active service does not switch to an unverified upload.

## HTTP routes

- `GET /health`: runtime model/version/engine diagnostics.
- `POST /ready`: authenticated Worker-to-Hostless readiness check.
- `POST /chat`: inference.
- `GET /admin/training/status`: training progress (authenticated).
- `POST /admin/training/start`: start or continue a staged 100K training cycle (authenticated).
- `POST /admin/training/cancel`: cancel at a safe training boundary (authenticated).
- `GET /admin/training/model`: download the compact model only after 100K steps (authenticated).
- `POST /admin/model-upload`: authenticated model activation from the Worker.

## Automated checks

GitHub Actions tests tokenizer preparation, the exact Brain 1.6 architecture/parameter count, checkpoint resume, quantized package round-trip, long-context attention gradients, Python service startup and HTTP smoke behavior. The older Brain 1.4 seed is checked as a separate legacy artifact, not used to initialize Brain 1.6.
