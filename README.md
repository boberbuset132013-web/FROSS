# FROSS 1.1 · Generative Brain 1.0

FROSS is a Russian-language experimental assistant with Google/email sign-in, saved chat history, a local rule-based engine, code templates, a Python service for normal requests, a C++ service for reasoning, and creator-password protected private learning memory.

## Main files
- [Web interface](public/index.html)
- [Firebase web configuration template](public/firebase-config.js)
- [Worker and API](worker.js)
- [FROSS Local Brain 0.6 JavaScript engine](fross-brain.js)
- [Python rule engine and tests 0.6](fross_brain_v05.py)
- [Trainable NumPy neural intent classifier](fross_neural_brain.py)
- [Python HTTP service](services/python-brain/server.py)
- [C++ planning engine](services/cpp-brain/main.cpp)
- [OpenAI-compatible text generation client](services/llm_client.py)
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

The JavaScript Worker can call a real hosted language model through an OpenAI-compatible API when the Cloudflare Secret `FROSS_LLM_API_KEY` is configured. Default settings target OpenRouter chat completions with the zero-priced `openrouter/free` model router; `FROSS_LLM_API_URL` and `FROSS_LLM_MODEL` can switch providers. Free models have rate limits and availability may vary. Normal conversational questions are not sent to web search by default when the model is configured; explicit requests for sources/current information trigger search and pass the results to the generator. Python and C++ services can also use `services/llm_client.py` when given the same model environment variables, and retain local algorithmic fallbacks on provider failure. The NumPy neural network remains a trained intent classifier using ReLU, softmax, cross-entropy and backpropagation; it is not the free-form text generator and is not trained on personal chats. The creator-only learning panel is unlocked by a Cloudflare Secret password. Account-private notes are stored in Firestore; when the `FROSS_GITHUB_TOKEN` Cloudflare Secret is configured, non-sensitive learning material is also written to `memory/knowledge/` as public, version-controlled topic files. A daily Cloudflare Cron Trigger searches and updates beginner material for JavaScript and Python, including code examples. The response limit is 25,000 characters, each user message is limited to 30,000 characters, and only the last five user turns with related assistant replies are sent as active chat context. Brain remains 0.5 while the product shell is FROSS 1.1. Follow `SETUP.md` to configure the services and secrets. Never store chats, passwords, tokens, or personal information in the public GitHub knowledge library. When connected, recent chat text is sent to the selected Render service; do not send passwords, tokens or private keys.
