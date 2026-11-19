# Traditional RAG vs. Agentic RAG (Gemini 3.5 Flash Lite)

Minimal reproducible comparison used for the tutorial post **"RAG Retrieves. Agentic RAG Thinks."**

## Requirements
- Python 3.12+
- `uv`
- A Google AI Studio API key in `GOOGLE_API_KEY` (or a local `.env` file) for the live benchmark

## Reproduce

Run offline unit tests only (no API key needed):

```bash
./reproduce.sh
```

Run both offline unit tests and the live `gemini-3.5-flash-lite` + `gemini-embedding-001` benchmark (3 queries x 3 trials):

```bash
GOOGLE_API_KEY="your-key" ./reproduce.sh
```
