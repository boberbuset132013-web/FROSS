# FROSS 1.0

FROSS is a Russian-language experimental assistant with Google/email sign-in, saved chat history, a local rule-based reasoning engine, limited code templates, and creator-only private learning memory.

## Main files
- [Web interface](public/index.html)
- [Firebase web configuration template](public/firebase-config.js)
- [Worker and API](worker.js)
- [FROSS Local Brain 0.4 JavaScript engine](fross-brain.js)
- [Python engine and tests 0.4](fross_brain_v04.py)
- [Python engine 0.3](fross_brain_v03.py)
- [Legacy Python entrypoint](fross_brain_v02.py)
- [Firestore security rules](firestore.rules)
- [Character and safety profile](FROSS_CHARACTER.md)
- [Setup guide](SETUP.md)
- [Wrangler configuration](wrangler.jsonc)

## First-time setup
Follow [SETUP.md](SETUP.md) before signing in. Firebase Web App values must be added to `public/firebase-config.js`, Google sign-in enabled, the Firestore database created, and `firestore.rules` published.

FROSS Local Brain 0.4 uses custom JavaScript rules and does not call a hosted language model. Web search is used when a question needs outside information; creator-only learning can save public page excerpts and source links into private Firestore memory. Saved notes are consulted before another search. Deploy with `npx wrangler deploy`.
