"""
ArXiv Research MCP Server
=========================
A production-grade MCP server for data scientists and ML researchers.
Provides tools to search ArXiv papers, find code implementations,
get SOTA benchmarks, fetch citations, and track trending research.

All APIs used are free and require no API keys:
  - ArXiv API (papers search & metadata)
  - Semantic Scholar API (citations & references)
  - Papers with Code API (code repos & benchmarks)

Author: Dr. Swapnil Morande
License: MIT
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================
# Server Initialization
# ============================================================

import os

SERVER_PORT = int(os.environ.get("PORT", 8000))

mcp = FastMCP(
    "arxiv_research_mcp",
    host="0.0.0.0",
    port=SERVER_PORT,
)

# ============================================================
# Constants
# ============================================================

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
PAPERS_WITH_CODE_API = "https://paperswithcode.com/api/v1"
HUGGINGFACE_API = "https://huggingface.co/api"

ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.IR": "Information Retrieval",
    "cs.RO": "Robotics",
    "cs.CR": "Cryptography and Security",
    "cs.DB": "Databases",
    "cs.DS": "Data Structures and Algorithms",
    "cs.SE": "Software Engineering",
    "stat.ML": "Statistics - Machine Learning",
    "stat.ME": "Statistics - Methodology",
    "stat.AP": "Statistics - Applications",
    "math.ST": "Mathematics - Statistics",
    "eess.SP": "Signal Processing",
    "q-bio.QM": "Quantitative Methods (Biology)",
    "q-fin.ST": "Statistical Finance",
}

HTTP_TIMEOUT = 30.0
USER_AGENT = "ArXivResearchMCP/1.0 (MCP Server for Data Scientists)"

# ============================================================
# Shared HTTP Client
# ============================================================


async def _get_client() -> httpx.AsyncClient:
    """Create an async HTTP client with standard headers."""
    return httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


# ============================================================
# Shared Utilities
# ============================================================


def _clean_text(text: str) -> str:
    """Remove extra whitespace and newlines from text."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_arxiv_id(id_url: str) -> str:
    """Extract ArXiv ID from full URL."""
    return id_url.split("/abs/")[-1].split("/")[-1].replace("v", "").rstrip("0123456789") if "/abs/" in id_url else id_url.split("/")[-1]


def _format_paper_markdown(paper: Dict[str, Any], index: int = 0) -> str:
    """Format a single paper as markdown."""
    lines = []
    prefix = f"### {index}. " if index > 0 else "### "
    lines.append(f"{prefix}{paper.get('title', 'Untitled')}")
    lines.append("")

    if paper.get("authors"):
        authors = paper["authors"]
        if len(authors) > 5:
            authors_str = ", ".join(authors[:5]) + f" ... (+{len(authors)-5} more)"
        else:
            authors_str = ", ".join(authors)
        lines.append(f"**Authors:** {authors_str}")

    if paper.get("arxiv_id"):
        lines.append(f"**ArXiv ID:** [{paper['arxiv_id']}](https://arxiv.org/abs/{paper['arxiv_id']})")
    if paper.get("published"):
        lines.append(f"**Published:** {paper['published']}")
    if paper.get("categories"):
        cat_names = [f"`{c}`" for c in paper["categories"][:5]]
        lines.append(f"**Categories:** {', '.join(cat_names)}")
    if paper.get("citation_count") is not None:
        lines.append(f"**Citations:** {paper['citation_count']}")
    if paper.get("abstract"):
        abstract = paper["abstract"][:500]
        if len(paper["abstract"]) > 500:
            abstract += "..."
        lines.append(f"\n> {abstract}")

    lines.append("")
    return "\n".join(lines)


def _format_papers_list(papers: List[Dict], total: int = 0, query: str = "") -> str:
    """Format a list of papers as markdown."""
    if not papers:
        return "No papers found matching your query."

    lines = []
    if query:
        lines.append(f"## Search Results for: *{query}*")
    lines.append(f"**Found {total or len(papers)} papers** (showing {len(papers)})\n")

    for i, paper in enumerate(papers, 1):
        lines.append(_format_paper_markdown(paper, i))

    return "\n".join(lines)


# ============================================================
# ArXiv API Helpers
# ============================================================


def _parse_arxiv_entry(entry: ET.Element) -> Dict[str, Any]:
    """Parse a single ArXiv API entry into a dict."""
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    title = entry.find("atom:title", ns)
    summary = entry.find("atom:summary", ns)
    published = entry.find("atom:published", ns)
    updated = entry.find("atom:updated", ns)
    id_elem = entry.find("atom:id", ns)

    authors = []
    for author in entry.findall("atom:author", ns):
        name = author.find("atom:name", ns)
        if name is not None and name.text:
            authors.append(name.text)

    categories = []
    for cat in entry.findall("atom:category", ns):
        term = cat.get("term", "")
        if term:
            categories.append(term)

    links = {}
    for link in entry.findall("atom:link", ns):
        rel = link.get("rel", "")
        href = link.get("href", "")
        link_type = link.get("type", "")
        if rel == "alternate":
            links["abstract"] = href
        elif link_type == "application/pdf":
            links["pdf"] = href

    arxiv_id = ""
    if id_elem is not None and id_elem.text:
        arxiv_id = id_elem.text.split("/abs/")[-1] if "/abs/" in id_elem.text else id_elem.text.split("/")[-1]

    comment = entry.find("arxiv:comment", ns)
    journal_ref = entry.find("arxiv:journal_ref", ns)
    doi = entry.find("arxiv:doi", ns)

    return {
        "arxiv_id": arxiv_id,
        "title": _clean_text(title.text) if title is not None and title.text else "Untitled",
        "abstract": _clean_text(summary.text) if summary is not None and summary.text else "",
        "authors": authors,
        "categories": categories,
        "published": published.text[:10] if published is not None and published.text else "",
        "updated": updated.text[:10] if updated is not None and updated.text else "",
        "links": links,
        "comment": _clean_text(comment.text) if comment is not None and comment.text else None,
        "journal_ref": journal_ref.text.strip() if journal_ref is not None and journal_ref.text else None,
        "doi": doi.text.strip() if doi is not None and doi.text else None,
    }


async def _search_arxiv(query: str, max_results: int = 10, start: int = 0, sort_by: str = "relevance", category: str = "") -> Dict[str, Any]:
    """Execute ArXiv API search with retry logic."""
    import asyncio as _asyncio

    search_query = query
    if category:
        search_query = f"cat:{category} AND all:{query}" if query else f"cat:{category}"

    sort_map = {
        "relevance": "relevance",
        "date": "submittedDate",
        "last_updated": "lastUpdatedDate",
    }

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_map.get(sort_by, "relevance"),
        "sortOrder": "descending",
    }

    last_error = None
    for attempt in range(3):
        async with await _get_client() as client:
            try:
                response = await client.get(ARXIV_API_BASE, params=params)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < 2:
                    await _asyncio.sleep(1.5 * (attempt + 1))
    else:
        return {"papers": [], "total": 0, "start": start, "max_results": max_results, "error": f"ArXiv API unavailable after 3 retries: {last_error}"}

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

    total_elem = root.find("opensearch:totalResults", ns)
    total = int(total_elem.text) if total_elem is not None and total_elem.text else 0

    papers = []
    for entry in root.findall("atom:entry", ns):
        paper = _parse_arxiv_entry(entry)
        if paper["title"] != "Untitled" or paper["abstract"]:
            papers.append(paper)

    return {"papers": papers, "total": total, "start": start, "max_results": max_results}


async def _get_arxiv_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single paper by ArXiv ID with retry."""
    import asyncio as _asyncio

    params = {"id_list": arxiv_id, "max_results": 1}

    for attempt in range(3):
        async with await _get_client() as client:
            try:
                response = await client.get(ARXIV_API_BASE, params=params)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError:
                if attempt < 2:
                    await _asyncio.sleep(1.5 * (attempt + 1))
    else:
        return None

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entries = root.findall("atom:entry", ns)
    if entries:
        return _parse_arxiv_entry(entries[0])
    return None


# ============================================================
# Semantic Scholar API Helpers
# ============================================================


async def _get_semantic_scholar_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper data from Semantic Scholar by ArXiv ID."""
    url = f"{SEMANTIC_SCHOLAR_API}/paper/ARXIV:{arxiv_id}"
    fields = "title,citationCount,influentialCitationCount,references.title,references.citationCount,citations.title,citations.citationCount,citations.year,tldr,fieldsOfStudy,publicationTypes,year"

    async with await _get_client() as client:
        try:
            response = await client.get(url, params={"fields": fields})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            return None


async def _search_semantic_scholar(query: str, limit: int = 10, year: str = "", fields_of_study: str = "") -> Optional[Dict[str, Any]]:
    """Search papers via Semantic Scholar."""
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,citationCount,year,authors,externalIds,tldr,fieldsOfStudy,openAccessPdf",
    }
    if year:
        params["year"] = year
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    async with await _get_client() as client:
        try:
            response = await client.get(url, params=params)
            if response.status_code == 429:
                return {"error": "Rate limited. Semantic Scholar free tier allows ~100 requests/5 min. Try again shortly."}
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            return {"error": str(e)}


# ============================================================
# Papers with Code API Helpers
# ============================================================


async def _find_code_for_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Find code repositories for a paper via Papers with Code."""
    url = f"{PAPERS_WITH_CODE_API}/papers/"

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=False) as client:
        try:
            response = await client.get(url, params={"arxiv_id": arxiv_id})
            if response.status_code in (301, 302, 404):
                return None
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return None

            paper = results[0]
            paper_id = paper.get("id", "")

            repos_url = f"{PAPERS_WITH_CODE_API}/papers/{paper_id}/repositories/"
            repos_response = await client.get(repos_url)
            repos = []
            if repos_response.status_code == 200:
                try:
                    repos = repos_response.json().get("results", [])
                except Exception:
                    pass

            return {"paper": paper, "repositories": repos}

        except (httpx.HTTPStatusError, httpx.TimeoutException, Exception):
            return None


async def _get_sota_benchmarks(task: str = "", dataset: str = "") -> Optional[Dict[str, Any]]:
    """Get SOTA benchmarks. Tries Papers with Code first, falls back to HuggingFace."""

    # Try PwC API first (no follow redirects to detect if API is down)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=False) as client:
        try:
            if task:
                url = f"{PAPERS_WITH_CODE_API}/tasks/"
                response = await client.get(url, params={"q": task})
            elif dataset:
                url = f"{PAPERS_WITH_CODE_API}/datasets/"
                response = await client.get(url, params={"q": dataset})
            else:
                return None

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    return data
        except Exception:
            pass

    # Fallback: use HuggingFace API for datasets
    if dataset:
        async with await _get_client() as client:
            try:
                response = await client.get(f"{HUGGINGFACE_API}/datasets", params={"search": dataset, "limit": 15, "sort": "downloads", "direction": -1})
                response.raise_for_status()
                datasets = response.json()
                return {
                    "source": "huggingface",
                    "count": len(datasets),
                    "results": [
                        {
                            "name": d.get("id", ""),
                            "description": d.get("description", "")[:200] if d.get("description") else "",
                            "downloads": d.get("downloads", 0),
                            "likes": d.get("likes", 0),
                            "tags": d.get("tags", [])[:10],
                            "url": f"https://huggingface.co/datasets/{d.get('id', '')}",
                        }
                        for d in datasets
                    ],
                }
            except Exception:
                pass

    # Fallback: use HuggingFace models API for task-based search
    if task:
        async with await _get_client() as client:
            try:
                response = await client.get(f"{HUGGINGFACE_API}/models", params={"search": task, "limit": 15, "sort": "downloads", "direction": -1})
                response.raise_for_status()
                models = response.json()
                return {
                    "source": "huggingface",
                    "count": len(models),
                    "results": [
                        {
                            "name": m.get("id", ""),
                            "pipeline_tag": m.get("pipeline_tag", ""),
                            "downloads": m.get("downloads", 0),
                            "likes": m.get("likes", 0),
                            "tags": m.get("tags", [])[:10],
                            "url": f"https://huggingface.co/{m.get('id', '')}",
                        }
                        for m in models
                    ],
                }
            except Exception:
                pass

    return None


# ============================================================
# Input Models
# ============================================================


class SortBy(str, Enum):
    RELEVANCE = "relevance"
    DATE = "date"
    LAST_UPDATED = "last_updated"


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class SearchPapersInput(BaseModel):
    """Input for searching ArXiv papers."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Search query for ArXiv papers (e.g., 'transformer attention mechanism', 'graph neural networks')",
        min_length=1,
        max_length=500,
    )
    category: Optional[str] = Field(
        default=None,
        description="ArXiv category filter (e.g., 'cs.LG', 'cs.CL', 'stat.ML'). See arxiv_list_categories for full list.",
    )
    max_results: int = Field(default=10, description="Maximum number of results to return", ge=1, le=50)
    sort_by: SortBy = Field(default=SortBy.RELEVANCE, description="Sort order: 'relevance', 'date', or 'last_updated'")
    start: int = Field(default=0, description="Pagination offset for results", ge=0)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'")


class GetPaperInput(BaseModel):
    """Input for fetching a specific paper."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    arxiv_id: str = Field(
        ...,
        description="ArXiv paper ID (e.g., '2301.07041', '2106.09685')",
        min_length=4,
        max_length=20,
    )
    include_citations: bool = Field(default=True, description="Include citation data from Semantic Scholar")
    include_code: bool = Field(default=True, description="Include code repositories from Papers with Code")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'")

    @field_validator("arxiv_id")
    @classmethod
    def clean_arxiv_id(cls, v: str) -> str:
        v = v.strip()
        if "arxiv.org" in v:
            v = v.split("/abs/")[-1].split("/")[-1]
        v = re.sub(r"v\d+$", "", v)
        return v


class GetCitationsInput(BaseModel):
    """Input for fetching citation data."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    arxiv_id: str = Field(..., description="ArXiv paper ID", min_length=4, max_length=20)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    @field_validator("arxiv_id")
    @classmethod
    def clean_arxiv_id(cls, v: str) -> str:
        v = v.strip()
        if "arxiv.org" in v:
            v = v.split("/abs/")[-1].split("/")[-1]
        v = re.sub(r"v\d+$", "", v)
        return v


class FindCodeInput(BaseModel):
    """Input for finding code implementations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    arxiv_id: str = Field(..., description="ArXiv paper ID to find code for", min_length=4, max_length=20)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    @field_validator("arxiv_id")
    @classmethod
    def clean_arxiv_id(cls, v: str) -> str:
        v = v.strip()
        if "arxiv.org" in v:
            v = v.split("/abs/")[-1].split("/")[-1]
        v = re.sub(r"v\d+$", "", v)
        return v


class TrendingPapersInput(BaseModel):
    """Input for getting trending papers."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    category: str = Field(
        default="cs.LG",
        description="ArXiv category (e.g., 'cs.LG', 'cs.CL', 'cs.CV', 'stat.ML')",
    )
    days_back: int = Field(default=7, description="Look back N days for recent papers", ge=1, le=30)
    max_results: int = Field(default=10, description="Maximum results to return", ge=1, le=30)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class BenchmarkInput(BaseModel):
    """Input for SOTA benchmark lookup."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: Optional[str] = Field(
        default=None,
        description="ML task to search (e.g., 'image classification', 'object detection', 'machine translation')",
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Dataset to search (e.g., 'ImageNet', 'COCO', 'SQuAD')",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    @field_validator("task")
    @classmethod
    def validate_query(cls, v: Optional[str], info) -> Optional[str]:
        dataset = info.data.get("dataset")
        if not v and not dataset:
            raise ValueError("Provide at least one of 'task' or 'dataset'")
        return v


class SemanticSearchInput(BaseModel):
    """Input for semantic paper search via Semantic Scholar."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Research topic or question", min_length=2, max_length=300)
    limit: int = Field(default=10, ge=1, le=100)
    year: Optional[str] = Field(default=None, description="Filter by year or range (e.g., '2024', '2023-2024')")
    fields_of_study: Optional[str] = Field(
        default=None,
        description="Filter by field (e.g., 'Computer Science', 'Mathematics', 'Medicine')",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


# ============================================================
# MCP Tools
# ============================================================


@mcp.tool(
    name="arxiv_search_papers",
    annotations={
        "title": "Search ArXiv Papers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def arxiv_search_papers(params: SearchPapersInput) -> str:
    """Search ArXiv for research papers by keyword, topic, or category.

    Searches the full ArXiv database of 2M+ papers. Supports filtering by category
    (cs.LG, cs.CL, stat.ML, etc.) and sorting by relevance, date, or last updated.
    Returns titles, authors, abstracts, ArXiv IDs, and links.

    Args:
        params (SearchPapersInput): Search parameters including:
            - query (str): Search query keywords
            - category (Optional[str]): ArXiv category filter
            - max_results (int): Max results 1-50
            - sort_by (SortBy): Sort order
            - start (int): Pagination offset
            - response_format (ResponseFormat): markdown or json

    Returns:
        str: Formatted paper results in markdown or JSON
    """
    result = await _search_arxiv(
        query=params.query,
        max_results=params.max_results,
        start=params.start,
        sort_by=params.sort_by.value,
        category=params.category or "",
    )

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    return _format_papers_list(result["papers"], result["total"], params.query)


@mcp.tool(
    name="arxiv_get_paper",
    annotations={
        "title": "Get Full Paper Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def arxiv_get_paper(params: GetPaperInput) -> str:
    """Get comprehensive details for a specific ArXiv paper by ID.

    Fetches paper metadata from ArXiv, optionally enriched with citation data
    from Semantic Scholar and code repositories from Papers with Code.

    Args:
        params (GetPaperInput): Parameters including:
            - arxiv_id (str): The ArXiv paper ID
            - include_citations (bool): Add Semantic Scholar citation data
            - include_code (bool): Add Papers with Code repos
            - response_format (ResponseFormat): markdown or json

    Returns:
        str: Complete paper details in markdown or JSON
    """
    paper = await _get_arxiv_paper(params.arxiv_id)
    if not paper:
        return f"Error: Paper with ArXiv ID '{params.arxiv_id}' not found. Check the ID format (e.g., '2301.07041')."

    enriched = dict(paper)

    if params.include_citations:
        ss_data = await _get_semantic_scholar_paper(params.arxiv_id)
        if ss_data and "error" not in ss_data:
            enriched["citation_count"] = ss_data.get("citationCount", 0)
            enriched["influential_citations"] = ss_data.get("influentialCitationCount", 0)
            enriched["tldr"] = ss_data.get("tldr", {}).get("text") if ss_data.get("tldr") else None
            enriched["fields_of_study"] = ss_data.get("fieldsOfStudy", [])

            top_citations = sorted(
                ss_data.get("citations", []),
                key=lambda x: x.get("citationCount", 0),
                reverse=True,
            )[:5]
            enriched["top_citing_papers"] = [{"title": c.get("title", ""), "citations": c.get("citationCount", 0), "year": c.get("year")} for c in top_citations]

            top_refs = sorted(
                ss_data.get("references", []),
                key=lambda x: x.get("citationCount", 0),
                reverse=True,
            )[:5]
            enriched["top_references"] = [{"title": r.get("title", ""), "citations": r.get("citationCount", 0)} for r in top_refs]

    if params.include_code:
        code_data = await _find_code_for_paper(params.arxiv_id)
        if code_data:
            repos = code_data.get("repositories", [])
            enriched["code_repositories"] = [
                {
                    "url": r.get("url", ""),
                    "stars": r.get("stars", 0),
                    "framework": r.get("framework", ""),
                    "is_official": r.get("is_official", False),
                }
                for r in repos[:5]
            ]

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(enriched, indent=2)

    # Build rich markdown
    lines = [f"# {enriched['title']}", ""]

    if enriched.get("tldr"):
        lines.append(f"**TL;DR:** {enriched['tldr']}")
        lines.append("")

    lines.append(f"**ArXiv ID:** [{enriched['arxiv_id']}](https://arxiv.org/abs/{enriched['arxiv_id']})")
    lines.append(f"**Published:** {enriched.get('published', 'N/A')} | **Updated:** {enriched.get('updated', 'N/A')}")

    if enriched.get("authors"):
        lines.append(f"**Authors:** {', '.join(enriched['authors'])}")
    if enriched.get("categories"):
        cats = [f"`{c}` ({ARXIV_CATEGORIES.get(c, '')})" if c in ARXIV_CATEGORIES else f"`{c}`" for c in enriched["categories"]]
        lines.append(f"**Categories:** {', '.join(cats)}")
    if enriched.get("fields_of_study"):
        lines.append(f"**Fields:** {', '.join(enriched['fields_of_study'])}")
    if enriched.get("doi"):
        lines.append(f"**DOI:** {enriched['doi']}")
    if enriched.get("journal_ref"):
        lines.append(f"**Journal:** {enriched['journal_ref']}")
    if enriched.get("comment"):
        lines.append(f"**Comment:** {enriched['comment']}")

    lines.append("")
    if enriched.get("links"):
        link_parts = []
        if "pdf" in enriched["links"]:
            link_parts.append(f"[PDF]({enriched['links']['pdf']})")
        if "abstract" in enriched["links"]:
            link_parts.append(f"[Abstract]({enriched['links']['abstract']})")
        lines.append(f"**Links:** {' | '.join(link_parts)}")

    if enriched.get("citation_count") is not None:
        lines.append("")
        lines.append("## Citation Impact")
        lines.append(f"**Total Citations:** {enriched['citation_count']} | **Influential:** {enriched.get('influential_citations', 0)}")

        if enriched.get("top_citing_papers"):
            lines.append("\n**Top Citing Papers:**")
            for cp in enriched["top_citing_papers"]:
                year_str = f" ({cp['year']})" if cp.get("year") else ""
                lines.append(f"  • {cp['title']}{year_str} — {cp['citations']} citations")

        if enriched.get("top_references"):
            lines.append("\n**Key References:**")
            for ref in enriched["top_references"]:
                lines.append(f"  • {ref['title']} — {ref['citations']} citations")

    if enriched.get("code_repositories"):
        lines.append("")
        lines.append("## Code Implementations")
        for repo in enriched["code_repositories"]:
            official = " ⭐ OFFICIAL" if repo.get("is_official") else ""
            framework = f" ({repo['framework']})" if repo.get("framework") else ""
            lines.append(f"  • [{repo['url']}]({repo['url']}) — {repo['stars']} stars{framework}{official}")

    lines.append("")
    lines.append("## Abstract")
    lines.append(enriched.get("abstract", "No abstract available."))

    return "\n".join(lines)


@mcp.tool(
    name="arxiv_get_citations",
    annotations={
        "title": "Get Paper Citations",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def arxiv_get_citations(params: GetCitationsInput) -> str:
    """Get citation data for an ArXiv paper via Semantic Scholar.

    Returns total citations, influential citations, top citing papers,
    key references, TL;DR summary, and fields of study.

    Args:
        params (GetCitationsInput): Parameters including:
            - arxiv_id (str): ArXiv paper ID
            - response_format (ResponseFormat): markdown or json

    Returns:
        str: Citation analysis in markdown or JSON
    """
    ss_data = await _get_semantic_scholar_paper(params.arxiv_id)
    if not ss_data:
        return f"Error: No citation data found for ArXiv ID '{params.arxiv_id}'. The paper may be too recent or not indexed by Semantic Scholar."

    if "error" in ss_data:
        return f"Error: {ss_data['error']}"

    result = {
        "arxiv_id": params.arxiv_id,
        "title": ss_data.get("title", ""),
        "citation_count": ss_data.get("citationCount", 0),
        "influential_citation_count": ss_data.get("influentialCitationCount", 0),
        "tldr": ss_data.get("tldr", {}).get("text") if ss_data.get("tldr") else None,
        "fields_of_study": ss_data.get("fieldsOfStudy", []),
        "year": ss_data.get("year"),
    }

    citations = sorted(ss_data.get("citations", []), key=lambda x: x.get("citationCount", 0), reverse=True)
    result["top_citations"] = [{"title": c.get("title", ""), "citations": c.get("citationCount", 0), "year": c.get("year")} for c in citations[:10]]

    references = sorted(ss_data.get("references", []), key=lambda x: x.get("citationCount", 0), reverse=True)
    result["top_references"] = [{"title": r.get("title", ""), "citations": r.get("citationCount", 0)} for r in references[:10]]

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    lines = [f"# Citation Analysis: {result['title']}", ""]
    lines.append(f"**ArXiv ID:** {params.arxiv_id} | **Year:** {result.get('year', 'N/A')}")
    lines.append(f"**Total Citations:** {result['citation_count']} | **Influential:** {result['influential_citation_count']}")

    if result.get("tldr"):
        lines.append(f"\n**TL;DR:** {result['tldr']}")
    if result.get("fields_of_study"):
        lines.append(f"**Fields:** {', '.join(result['fields_of_study'])}")

    if result["top_citations"]:
        lines.append("\n## Top Citing Papers")
        for i, c in enumerate(result["top_citations"], 1):
            year_str = f" ({c['year']})" if c.get("year") else ""
            lines.append(f"{i}. {c['title']}{year_str} — {c['citations']} citations")

    if result["top_references"]:
        lines.append("\n## Key References (most cited)")
        for i, r in enumerate(result["top_references"], 1):
            lines.append(f"{i}. {r['title']} — {r['citations']} citations")

    return "\n".join(lines)


@mcp.tool(
    name="arxiv_find_code",
    annotations={
        "title": "Find Code for Paper",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def arxiv_find_code(params: FindCodeInput) -> str:
    """Find code implementations and GitHub repos for an ArXiv paper.

    Searches Papers with Code to find official and community implementations
    of a research paper. Returns repo URLs, star counts, and frameworks used.

    Args:
        params (FindCodeInput): Parameters including:
            - arxiv_id (str): ArXiv paper ID
            - response_format (ResponseFormat): markdown or json

    Returns:
        str: Code repository information in markdown or JSON
    """
    code_data = await _find_code_for_paper(params.arxiv_id)

    if not code_data:
        return f"No code repositories found for ArXiv ID '{params.arxiv_id}' on Papers with Code. The paper may not have public implementations yet."

    paper_info = code_data.get("paper", {})
    repos = code_data.get("repositories", [])

    result = {
        "arxiv_id": params.arxiv_id,
        "paper_title": paper_info.get("title", ""),
        "paper_url_pwc": paper_info.get("url_abs", ""),
        "repository_count": len(repos),
        "repositories": [
            {
                "url": r.get("url", ""),
                "stars": r.get("stars", 0),
                "framework": r.get("framework", ""),
                "is_official": r.get("is_official", False),
                "description": r.get("description", ""),
            }
            for r in repos
        ],
    }

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    lines = [f"# Code for: {result['paper_title']}", ""]
    lines.append(f"**ArXiv ID:** {params.arxiv_id}")
    lines.append(f"**Papers with Code:** {result['paper_url_pwc']}")
    lines.append(f"**Total Repositories:** {result['repository_count']}")

    if repos:
        lines.append("\n## Repositories")
        sorted_repos = sorted(result["repositories"], key=lambda x: (-x.get("is_official", False), -x.get("stars", 0)))
        for r in sorted_repos:
            official = "⭐ OFFICIAL " if r["is_official"] else ""
            framework = f" [{r['framework']}]" if r["framework"] else ""
            lines.append(f"  • {official}[{r['url']}]({r['url']}) — {r['stars']} stars{framework}")
            if r.get("description"):
                lines.append(f"    {r['description'][:200]}")
    else:
        lines.append("\nNo repositories found yet.")

    return "\n".join(lines)


@mcp.tool(
    name="arxiv_trending_papers",
    annotations={
        "title": "Get Trending Papers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def arxiv_trending_papers(params: TrendingPapersInput) -> str:
    """Get recently submitted papers in an ArXiv category, sorted by date.

    Fetches the most recent papers submitted to a specific ArXiv category.
    Useful for tracking new research in ML, NLP, CV, and other fields.

    Args:
        params (TrendingPapersInput): Parameters including:
            - category (str): ArXiv category (e.g., 'cs.LG', 'cs.CL')
            - days_back (int): Look back N days (1-30)
            - max_results (int): Max results (1-30)
            - response_format (ResponseFormat): markdown or json

    Returns:
        str: Recent papers in markdown or JSON
    """
    result = await _search_arxiv(
        query="",
        max_results=params.max_results,
        sort_by="date",
        category=params.category,
    )

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    cat_name = ARXIV_CATEGORIES.get(params.category, params.category)
    lines = [f"## Trending in {cat_name} (`{params.category}`)", ""]
    lines.append(f"**Showing {len(result['papers'])} most recent papers**\n")

    for i, paper in enumerate(result["papers"], 1):
        lines.append(_format_paper_markdown(paper, i))

    return "\n".join(lines)


@mcp.tool(
    name="arxiv_get_benchmarks",
    annotations={
        "title": "Get SOTA Benchmarks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def arxiv_get_benchmarks(params: BenchmarkInput) -> str:
    """Look up state-of-the-art benchmarks for ML tasks or datasets.

    Searches Papers with Code for benchmark leaderboards. Provide either a task
    name (e.g., 'image classification') or dataset name (e.g., 'ImageNet').

    Args:
        params (BenchmarkInput): Parameters including:
            - task (Optional[str]): ML task name
            - dataset (Optional[str]): Dataset name
            - response_format (ResponseFormat): markdown or json

    Returns:
        str: Benchmark/task/dataset information in markdown or JSON
    """
    data = await _get_sota_benchmarks(task=params.task or "", dataset=params.dataset or "")

    if not data:
        return "Error: Could not fetch benchmark data. Please try a different task or dataset name."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, indent=2)

    results = data.get("results", [])
    if not results:
        query = params.task or params.dataset
        return f"No benchmarks found for '{query}'. Try a broader term (e.g., 'classification' instead of 'binary classification')."

    source = data.get("source", "paperswithcode")
    search_type = "Task" if params.task else "Dataset"
    query = params.task or params.dataset
    lines = [f"## {search_type} Search: *{query}*", ""]
    lines.append(f"**Source:** {source} | **Found {data.get('count', len(results))} results**\n")

    for item in results[:15]:
        name = item.get("name", "Unnamed")
        desc = item.get("description", "")[:200]

        lines.append(f"### {name}")

        if source == "huggingface":
            if item.get("downloads"):
                lines.append(f"  • Downloads: {item['downloads']:,}")
            if item.get("likes"):
                lines.append(f"  • Likes: {item['likes']}")
            if item.get("pipeline_tag"):
                lines.append(f"  • Pipeline: {item['pipeline_tag']}")
            if item.get("tags"):
                lines.append(f"  • Tags: {', '.join(item['tags'][:8])}")
            if item.get("url"):
                lines.append(f"  • [View on HuggingFace]({item['url']})")
        else:
            if item.get("paper_count"):
                lines.append(f"  • Papers: {item['paper_count']}")
            if item.get("num_papers"):
                lines.append(f"  • Papers: {item['num_papers']}")
            if item.get("url"):
                lines.append(f"  • [View on Papers with Code]({item['url']})")

        if desc:
            lines.append(f"  • {desc}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="arxiv_semantic_search",
    annotations={
        "title": "Semantic Paper Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def arxiv_semantic_search(params: SemanticSearchInput) -> str:
    """Search papers using Semantic Scholar's semantic search engine.

    Unlike keyword-based ArXiv search, this uses semantic understanding to find
    relevant papers. Supports filtering by year and field of study. Returns
    citation counts, TL;DR summaries, and open access PDF links.

    Args:
        params (SemanticSearchInput): Parameters including:
            - query (str): Research topic or question
            - limit (int): Max results (1-100)
            - year (Optional[str]): Year filter (e.g., '2024', '2023-2024')
            - fields_of_study (Optional[str]): Field filter
            - response_format (ResponseFormat): markdown or json

    Returns:
        str: Semantically matched papers in markdown or JSON
    """
    data = await _search_semantic_scholar(
        query=params.query,
        limit=params.limit,
        year=params.year or "",
        fields_of_study=params.fields_of_study or "",
    )

    if not data or "error" in data:
        error_msg = data.get("error", "Unknown error") if data else "No response"
        return f"Error: {error_msg}"

    papers = data.get("data", [])

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, indent=2)

    if not papers:
        return f"No papers found for '{params.query}'. Try broader terms."

    lines = [f"## Semantic Search: *{params.query}*", ""]
    lines.append(f"**Found {data.get('total', len(papers))} papers** (showing {len(papers)})\n")

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        citations = paper.get("citationCount", 0)
        year = paper.get("year", "")
        authors = paper.get("authors", [])
        tldr = paper.get("tldr", {})
        ext_ids = paper.get("externalIds", {})
        pdf_url = paper.get("openAccessPdf", {})

        lines.append(f"### {i}. {title}")
        if authors:
            auth_names = [a.get("name", "") for a in authors[:5]]
            if len(authors) > 5:
                auth_names.append(f"(+{len(authors)-5} more)")
            lines.append(f"**Authors:** {', '.join(auth_names)}")

        meta = []
        if year:
            meta.append(f"Year: {year}")
        meta.append(f"Citations: {citations}")
        if ext_ids and ext_ids.get("ArXiv"):
            meta.append(f"ArXiv: [{ext_ids['ArXiv']}](https://arxiv.org/abs/{ext_ids['ArXiv']})")
        lines.append(f"**{' | '.join(meta)}**")

        if tldr and tldr.get("text"):
            lines.append(f"> {tldr['text']}")
        if pdf_url and pdf_url.get("url"):
            lines.append(f"[Open Access PDF]({pdf_url['url']})")

        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="arxiv_list_categories",
    annotations={
        "title": "List ArXiv Categories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def arxiv_list_categories() -> str:
    """List all supported ArXiv categories relevant to data science and ML.

    Returns the category codes and their full names, useful for filtering
    searches with arxiv_search_papers and arxiv_trending_papers.

    Returns:
        str: Formatted list of ArXiv categories
    """
    lines = ["## ArXiv Categories (Data Science & ML)", ""]
    for code, name in sorted(ARXIV_CATEGORIES.items()):
        lines.append(f"  • `{code}` — {name}")
    lines.append("")
    lines.append("Use these codes with `arxiv_search_papers` or `arxiv_trending_papers` to filter by category.")
    return "\n".join(lines)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
