FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY arxiv_research_mcp.py .

ENV PORT=8000
EXPOSE ${PORT}

CMD ["python", "arxiv_research_mcp.py"]
