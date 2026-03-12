# ArXiv Research MCP Server 🔬

**The MCP server every data scientist needs.** Search 2M+ ArXiv papers, find code implementations, track citations, get SOTA benchmarks — all from your AI assistant.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/smorande/arxiv-research-mcp)

## Why This Server?

Data scientists spend ~30% of their time searching for papers, checking if code exists, comparing benchmarks. This MCP server brings all of that into your LLM workflow with zero API keys required.

## Tools Available

| Tool | What it does |
|------|-------------|
| `arxiv_search_papers` | Full-text search across 2M+ ArXiv papers with category filtering |
| `arxiv_get_paper` | Complete paper details with citations + code repos enrichment |
| `arxiv_get_citations` | Citation analysis via Semantic Scholar (citing papers, references) |
| `arxiv_find_code` | Find GitHub repos implementing a paper (Papers with Code) |
| `arxiv_trending_papers` | Latest papers in any ArXiv category |
| `arxiv_get_benchmarks` | SOTA benchmark lookup by task or dataset |
| `arxiv_semantic_search` | Semantic search via Semantic Scholar (better than keyword search) |
| `arxiv_list_categories` | List all supported ArXiv categories |

## Quick Start

### Option 1: Connect directly (Streamable HTTP)

Add to your Claude Desktop / MCP client config:

```json
{
  "mcpServers": {
    "arxiv-research": {
      "type": "url",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Option 2: Run locally

```bash
pip install -r requirements.txt
python arxiv_research_mcp.py
```

### Option 3: Docker

```bash
docker build -t arxiv-research-mcp .
docker run -p 8000:8000 arxiv-research-mcp
```

### Option 4: Using uv (recommended)

```bash
uv run --with "mcp[cli]" --with httpx --with pydantic arxiv_research_mcp.py
```

## Example Queries

Once connected, ask your AI assistant:

- *"Search ArXiv for papers on LoRA fine-tuning from 2024"*
- *"Get full details for paper 2106.09685 including citations and code"*
- *"Find code implementations for the Mamba architecture paper"*
- *"What are the trending papers in cs.CL this week?"*
- *"Show SOTA benchmarks for object detection"*
- *"Semantic search for papers on retrieval augmented generation"*

## APIs Used (All Free, No Keys)

| API | Rate Limit | Data |
|-----|-----------|------|
| ArXiv API | 1 req/3s | 2M+ papers, full metadata |
| Semantic Scholar | ~100 req/5min | Citations, references, TL;DR |
| Papers with Code | Fair use | Code repos, benchmarks, tasks |

## Deployment

### Railway / Render / Fly.io

The server runs on port 8000 with streamable HTTP transport. Deploy the Dockerfile to any container hosting platform. No environment variables or API keys needed.

### Permanent self-hosting

```bash
docker compose up -d
```

## License

MIT — use it, fork it, make it better.

## Author

Dr. Swapnil Morande — Principal AI Architect
