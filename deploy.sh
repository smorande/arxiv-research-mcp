#!/bin/bash
# ============================================================
# ArXiv Research MCP Server — GitHub + Railway Deploy Script
# Run this on your Mac terminal. Takes ~3 minutes.
# ============================================================

set -e

echo "============================================================"
echo "ArXiv Research MCP Server — Deploy to GitHub + Railway"
echo "============================================================"

# ---- Step 1: Check prerequisites ----
echo ""
echo "Step 1: Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo "❌ git not found. Install: brew install git"
    exit 1
fi
echo "  ✅ git found"

if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI not found. Install: brew install gh"
    echo "  Then run: gh auth login"
    exit 1
fi
echo "  ✅ gh CLI found"

# Check gh auth
if ! gh auth status &> /dev/null; then
    echo "❌ Not logged into GitHub. Run: gh auth login"
    exit 1
fi
echo "  ✅ GitHub authenticated"

# ---- Step 2: Create GitHub repo ----
echo ""
echo "Step 2: Creating GitHub repo..."

REPO_NAME="arxiv-research-mcp"

# Check if repo already exists
if gh repo view "$REPO_NAME" &> /dev/null 2>&1; then
    echo "  ⚠️ Repo '$REPO_NAME' already exists. Pushing to existing repo."
else
    gh repo create "$REPO_NAME" \
        --public \
        --description "MCP server for data scientists — search ArXiv papers, find code, get citations, track SOTA benchmarks. Zero API keys needed." \
        --homepage "https://arxiv.org" \
        --clone=false
    echo "  ✅ GitHub repo created: $REPO_NAME"
fi

# Get the GitHub username
GH_USER=$(gh api user -q .login)
REPO_URL="https://github.com/$GH_USER/$REPO_NAME.git"
echo "  Repo URL: $REPO_URL"

# ---- Step 3: Push code ----
echo ""
echo "Step 3: Pushing code to GitHub..."

git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git push -u origin main --force
echo "  ✅ Code pushed to GitHub"

# ---- Step 4: Railway deploy ----
echo ""
echo "Step 4: Deploying to Railway..."

if ! command -v railway &> /dev/null; then
    echo ""
    echo "⚠️ Railway CLI not found. Install it:"
    echo "  npm install -g @railway/cli"
    echo "  railway login"
    echo ""
    echo "Then run these commands from the project directory:"
    echo "  cd $(pwd)"
    echo "  railway init"
    echo "  railway up"
    echo "  railway domain   # generates a public URL"
    echo ""
    echo "Your MCP endpoint will be: https://<your-domain>.up.railway.app/mcp"
    echo ""
    echo "GitHub repo is live at: $REPO_URL"
    exit 0
fi

# Check railway auth
if ! railway whoami &> /dev/null 2>&1; then
    echo "❌ Not logged into Railway. Run: railway login"
    exit 1
fi
echo "  ✅ Railway authenticated"

# Create Railway project and deploy
railway init --name "$REPO_NAME" 2>/dev/null || true
railway up --detach

echo ""
echo "  ✅ Deployed to Railway!"
echo ""
echo "Step 5: Generating public domain..."
railway domain

echo ""
echo "============================================================"
echo "🎉 DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "GitHub Repo:  $REPO_URL"
echo "MCP Endpoint: https://<your-railway-domain>.up.railway.app/mcp"
echo ""
echo "Connect to Claude Desktop / any MCP client:"
echo '{'
echo '  "mcpServers": {'
echo '    "arxiv-research": {'
echo '      "type": "url",'
echo '      "url": "https://<your-railway-domain>.up.railway.app/mcp"'
echo '    }'
echo '  }'
echo '}'
echo ""
