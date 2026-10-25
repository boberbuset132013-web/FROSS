# FROSS 1.1 · Transformer Brain 1.0

FROSS is a Russian-language experimental assistant with a decoder-only Transformer implemented and trained from scratch, an original corpus and tokenizer, a Python generation service, saved chat history, and private learning memory.

## Main files
- [Web interface](public/index.html)
- [Firebase web configuration template](public/firebase-config.js)
- [FROSS Worker and request router](worker.js)
- [From-scratch Transformer and BPE tokenizer](fross_transformer.py)
- [Original training corpus](data/fross_corpus.jsonl)
- [Transformer architecture tests](tests/test_fross_transformer.py)
- [Python HTTP service](services/python-brain/server.py)
- [Render blueprint for one Python service](render.yaml)
- [Firestore security rules](firestore.rules)
- [Memory design overview](memory/README.md)
- [Shared GitHub knowledge index](memory/knowledge/index.json)
- [Shared GitHub knowledge folder](memory/knowledge/)
- [Character and safety profile](FROSS_CHARACTER.md)
- [Setup guide](SETUP.md)
- [Wrangler configuration](wrangler.jsonc)

## First-time setup
Follow [SETUP.md](SETUP.md) before signing in. Firebase Web App values must be added to `public/firebase-config.js`, Google sign-in enabled, the Firestore database created, and `firestore.rules` published.

The response generator is our own small decoder-only Transformer in `fross_transformer.py`: a corpus-trained BPE tokenizer, token and positional embeddings, causal multi-head self-attention, feed-forward blocks, residual connections, LayerNorm, and next-token sampling. Weights are initialized randomly, trained locally during the single Python service's Docker build on the original Russian corpus in `data/fross_corpus.jsonl`, and saved as a checkpoint. No pretrained language model, downloaded weights, or hosted LLM API is used. The Worker routes both normal and reasoning mode to that same Python service; reasoning mode only changes the prompt and token budget. If the checkpoint is missing, the service reports an error instead of silently replacing a substantive answer with a canned template. Web search is opt-in for requests asking for recent facts or sources. The model is still a small experiment: 198 original training pairs and 1800 CPU training steps, with checkpoint selection by validation loss cannot provide the knowledge or reliability of a large-scale language model; quality must be measured against held-out examples and improved with more data and compute. The creator-only learning panel is unlocked by a Cloudflare Secret password. Account-private notes are stored in Firestore; when the `FROSS_GITHUB_TOKEN` Cloudflare Secret is configured, non-sensitive learning material is also written to `memory/knowledge/` as public, version-controlled topic files. A daily Cloudflare Cron Trigger searches and updates beginner material for JavaScript and Python, including code examples. The response limit is 25,000 characters, each user message is limited to 30,000 characters, and only the last five user turns with related assistant replies are sent as active chat context. The Worker reports Brain 1.0 while the product shell is FROSS 1.1. Follow `SETUP.md` to configure the services and secrets. Never store chats, passwords, tokens, or personal information in the public GitHub knowledge library. When connected, recent chat text is sent to the configured Python generation service; do not send passwords, tokens or private keys.
