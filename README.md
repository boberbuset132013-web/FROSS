# FROSS 1.0

FROSS is a Russian-language experimental assistant with Google/email sign-in, saved chat history, web search, a rule-based local reasoning engine, and a 60-user-message limit per chat.

## Main files
- [Web interface](public/index.html)
- [Firebase web configuration template](public/firebase-config.js)
- [Worker and API](worker.js)
- [FROSS Local Brain rules and templates](fross-brain.js)
- [Firestore security rules](firestore.rules)
- [Character and safety profile](FROSS_CHARACTER.md)
- [Setup guide](SETUP.md)
- [Wrangler configuration](wrangler.jsonc)

## First-time setup
Follow [SETUP.md](SETUP.md) before signing in. Firebase Web App values must be added to `public/firebase-config.js`, Google sign-in enabled, the Firestore database created, and `firestore.rules` published.

FROSS Local Brain 0.1 uses custom JavaScript rules and does not call a hosted language model. Web search is still used to gather links and short snippets. Deploy with `npx wrangler deploy`.
