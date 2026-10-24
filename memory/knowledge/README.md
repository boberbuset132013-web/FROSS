# Shared FROSS knowledge library

This folder stores researched learning material in version-controlled JSON files.

- `index.json` — small searchable catalog of topics, summaries, key points, keywords, source counts, and update timestamps.
- `<topic-slug>.json` — topic records with summaries, key points, keywords, URLs, source excerpts, and source scores.

The Worker updates the topic document and then the index when creator-only training succeeds. At chat time it searches the public index, retrieves the best matching topic files, and combines those records with the account's private Firestore notes and current web results.

This folder is public. Only store non-sensitive educational material, not personal conversation history or secrets.
