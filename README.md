# FROSS 1.0

FROSS is a Russian-language AI assistant with Google sign-in, saved chat history, web search, and a 60-user-message limit per chat.

## Main files
- [Web interface](public/index.html)
- [Firebase web configuration template](public/firebase-config.js)
- [Worker and AI API](worker.js)
- [Firestore security rules](firestore.rules)
- [Character and safety profile](FROSS_CHARACTER.md)
- [Setup guide](SETUP.md)
- [Wrangler configuration](wrangler.jsonc)

## First-time setup
Follow [SETUP.md](SETUP.md) before signing in. Firebase Web App values must be added to `public/firebase-config.js`, Google sign-in enabled, the Firestore database created, and `firestore.rules` published.

Cloudflare Workers AI is configured using the AI binding in `wrangler.jsonc`. Deploy with `npx wrangler deploy`.
