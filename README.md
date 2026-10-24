# FROSS 1.0

FROSS is a Russian-language experimental assistant with Google/email sign-in, saved chat history, a local rule-based engine, code templates, a Python service for normal requests, a C++ service for reasoning, and creator-only private learning memory.

## Main files
- [Web interface](public/index.html)
- [Firebase web configuration template](public/firebase-config.js)
- [Worker and API](worker.js)
- [FROSS Local Brain 0.4 JavaScript engine](fross-brain.js)
- [Python engine and tests 0.4](fross_brain_v04.py)
- [Python HTTP service](services/python-brain/server.py)
- [C++ reasoning engine](services/cpp-brain/main.cpp)
- [C++ HTTP service](services/cpp-brain/server.py)
- [Render blueprint for two free services](render.yaml)
- [Python engine 0.3](fross_brain_v03.py)
- [Legacy Python entrypoint](fross_brain_v02.py)
- [Firestore security rules](firestore.rules)
- [Character and safety profile](FROSS_CHARACTER.md)
- [Setup guide](SETUP.md)
- [Wrangler configuration](wrangler.jsonc)

## First-time setup
Follow [SETUP.md](SETUP.md) before signing in. Firebase Web App values must be added to `public/firebase-config.js`, Google sign-in enabled, the Firestore database created, and `firestore.rules` published.

JavaScript dispatches ordinary requests to the Python HTTP service and deep-reasoning requests to the C++ HTTP service when their URLs and shared secret are configured in Cloudflare. If a service is unavailable or cannot answer, the Worker falls back to its local JavaScript brain. No hosted language model is called. Web search is reserved for questions that appear to need external facts. Creator-only learning can save public page excerpts and source links in private Firestore memory. Follow `SETUP.md` to deploy the two free service instances and configure secrets.
