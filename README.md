# Traditional RAG vs. Agentic RAG (Gemini 3.5 Flash Lite)

Reproducible side-by-side comparison used for the tutorial post **"RAG Retrieves. Agentic RAG Thinks."**

Compares **Traditional RAG (`k=2` and `k=4`)** against **Agentic RAG (`k=2`, 2 tools + `recursion_limit=10`)** across a 12-document engineering handbook and 5 query archetypes (3 trials each) using `gemini-3.5-flash-lite` and `gemini-embedding-001`.

## Requirements
- Python 3.12+
- `uv`
- A Google AI Studio API key in `GOOGLE_API_KEY` (or a local `.env` / `DOTENV_FILE`) for the live benchmark

## Reproduce

Run offline deterministic unit tests only (no API key needed):

```bash
./reproduce.sh
```

Run both offline unit tests and the live `gemini-3.5-flash-lite` + `gemini-embedding-001` benchmark:

```bash
GOOGLE_API_KEY="your-key" ./reproduce.sh
```
