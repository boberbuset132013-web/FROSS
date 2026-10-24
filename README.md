# FROSS 1.1

FROSS is a Russian-language experimental assistant with Google/email sign-in, saved chat history, a local rule-based engine, code templates, a Python service for normal requests, a C++ service for reasoning, and creator-password protected private learning memory.

## Main files
- [Web interface](public/index.html)
- [Firebase web configuration template](public/firebase-config.js)
- [Worker and API](worker.js)
- [FROSS Local Brain 0.5 JavaScript engine](fross-brain.js)
- [Python engine and tests 0.5](fross_brain_v05.py)
- [Python HTTP service](services/python-brain/server.py)
- [C++ reasoning engine](services/cpp-brain/main.cpp)
- [C++ HTTP service](services/cpp-brain/server.py)
- [Render blueprint for two free services](render.yaml)
- [Python engine 0.3](fross_brain_v03.py)
- [Legacy Python entrypoint](fross_brain_v02.py)
- [Firestore security rules](firestore.rules)
- [Memory design overview](memory/README.md)
- [Shared GitHub knowledge index](memory/knowledge/index.json)
- [Shared GitHub knowledge folder](memory/knowledge/)
- [Character and safety profile](FROSS_CHARACTER.md)
- [Setup guide](SETUP.md)
- [Wrangler configuration](wrangler.jsonc)

## First-time setup
Follow [SETUP.md](SETUP.md) before signing in. Firebase Web App values must be added to `public/firebase-config.js`, Google sign-in enabled, the Firestore database created, and `firestore.rules` published.

JavaScript dispatches ordinary requests to the Python HTTP service and deep-reasoning requests to the C++ HTTP service when their URLs and shared secret are configured in Cloudflare. If a service is unavailable or cannot answer, the Worker falls back to its local JavaScript brain. No hosted language model is called. Brain 0.5 blends saved knowledge with fresh search results for information queries. The creator-only learning panel is unlocked by a Cloudflare Secret password. Account-private notes are stored in Firestore; when the `FROSS_GITHUB_TOKEN` Cloudflare Secret is configured, non-sensitive learning material is also written to `memory/knowledge/` as public, version-controlled topic files. A daily Cloudflare Cron Trigger searches and updates beginner material for JavaScript and Python, including code examples. The response limit is 25,000 characters, each user message is limited to 30,000 characters, and only the last five user turns with related assistant replies are sent as active chat context. Brain remains 0.5 while the product shell is FROSS 1.1. Follow `SETUP.md` to configure the services and secrets. Never store chats, passwords, tokens, or personal information in the public GitHub knowledge library. When connected, recent chat text is sent to the selected Render service; do not send passwords, tokens or private keys.
