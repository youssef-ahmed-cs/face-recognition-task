#!/bin/bash

# Face Recognition - Vercel Deployment Script
# This script automates the deployment process

set -e

echo "🚀 Face Recognition - Deployment Script"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}1. Checking prerequisites...${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed${NC}"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ Node.js/npm is not installed${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites installed${NC}"
echo ""

# Build frontend
echo -e "${BLUE}2. Building frontend...${NC}"
mkdir -p public
cp src/client.html public/index.html
echo -e "${GREEN}✅ Frontend built${NC}"
echo ""

# Check if Git repo exists
if [ ! -d ".git" ]; then
    echo -e "${BLUE}3. Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit - Face Recognition API"
    echo -e "${GREEN}✅ Git repository initialized${NC}"
fi

echo ""
echo -e "${YELLOW}📋 Deployment Options:${NC}"
echo "1. Vercel (Frontend only)"
echo "2. Railway (Frontend + Backend)"
echo "3. Render (Frontend + Backend)"
echo "4. Docker (Local)"
echo ""
read -p "Choose deployment option (1-4): " choice

case $choice in
    1)
        echo -e "${BLUE}Deploying to Vercel (Frontend)...${NC}"
        if ! command -v vercel &> /dev/null; then
            echo "Installing Vercel CLI..."
            npm install -g vercel
        fi
        vercel
        echo -e "${GREEN}✅ Vercel deployment complete${NC}"
        ;;
    
    2)
        echo -e "${BLUE}Deploying to Railway...${NC}"
        if ! command -v railway &> /dev/null; then
            echo "Installing Railway CLI..."
            npm install -g @railway/cli
        fi
        railway login
        railway up
        echo -e "${GREEN}✅ Railway deployment complete${NC}"
        ;;
    
    3)
        echo -e "${BLUE}Deploying to Render...${NC}"
        echo "1. Go to https://render.com"
        echo "2. Click 'New +' → 'Web Service'"
        echo "3. Connect your GitHub repository"
        echo "4. Configure:"
        echo "   - Runtime: Python 3.9"
        echo "   - Build Command: pip install -r requirements.txt"
        echo "   - Start Command: python src/app.py"
        echo "5. Deploy"
        ;;
    
    4)
        echo -e "${BLUE}Building Docker image...${NC}"
        docker build -t face-recognition-api .
        echo -e "${GREEN}✅ Docker image built${NC}"
        echo ""
        echo "To run locally:"
        echo "docker run -p 8000:8000 face-recognition-api"
        ;;
    
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Deployment preparation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}📚 Next steps:${NC}"
echo "1. Visit VERCEL_DEPLOYMENT.md for detailed instructions"
echo "2. Configure environment variables on your hosting platform"
echo "3. Test your API at: http://localhost:8000/docs (locally)"
echo ""
