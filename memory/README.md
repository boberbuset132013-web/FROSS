# FROSS memory storage

The project separates two kinds of memory:

- `memory/knowledge/` in this public GitHub repository is the **shared, public knowledge library**. The creator-only training endpoint writes one JSON file per topic and updates `index.json`. FROSS can retrieve relevant entries from this library for later answers.
- Firebase Cloud Firestore `users/{uid}/knowledge` remains the **private per-account memory** used by the signed-in user's training panel.

Do not put chats, passwords, access tokens, personal details, or private notes into the GitHub library. Every commit under this folder is public and part of the repository history.

Runtime GitHub writes are disabled until a Cloudflare Worker Secret named `FROSS_GITHUB_TOKEN` is configured. It should be a fine-grained token restricted to this repository with Contents read/write permission. Never commit the token.
