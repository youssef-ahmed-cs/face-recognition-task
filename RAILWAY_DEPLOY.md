# Railway Deployment Guide (No CI/CD)

## 📋 Prerequisites

- Node.js and npm installed
- GitHub account with your repo pushed
- Git installed

---

## 🚀 Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

Or use Homebrew (Mac):
```bash
brew install railway
```

---

## 🔐 Step 2: Login to Railway

```bash
railway login
```

This opens your browser to authenticate. Follow the prompts.

---

## 📂 Step 3: Deploy from Repository

Navigate to your project directory:
```bash
cd "d:\4- Simple Face Recognition"
```

Create a new Railway project:
```bash
railway init
```

Select or create a new project when prompted.

---

## 🚀 Step 4: Deploy Your App

```bash
railway up
```

Railway will:
1. Detect your Python project
2. Install dependencies from `requirements.txt`
3. Run `python src/fastapi-deploy/app.py`
4. Assign you a public URL

---

## 🌐 Step 5: Get Your Backend URL

After deployment completes, Railway provides a public URL. Copy it:

```
https://your-app-name.railway.app
```

---

## 📝 Step 6: Update Frontend API URL

Edit `src/client.html`:

```html
<!-- Find this line (around line 123): -->
const API_URL = 'http://localhost:8000';

<!-- Change to: -->
const API_URL = 'https://your-app-name.railway.app';
```

---

## 🧪 Testing

Test your backend at:
```
https://your-app-name.railway.app/health
```

Should return: `{"status":"ok"}`

---

## 📊 Monitor Your Deployment

View logs:
```bash
railway logs
```

View dashboard:
```bash
railway open
```

---

## 🔄 Redeployments

Make changes locally, push to GitHub, then:

```bash
# Option 1: Manual redeploy
railway up

# Option 2: From Railway dashboard
# Settings → Redeploy on push → Enable
```

---

## 🆘 Troubleshooting

### "Port 8000 already in use"
```bash
# Kill the process
taskkill /F /IM python.exe

# Or redeploy
railway up
```

### "Module not found"
```bash
# Ensure requirements.txt is up to date
pip freeze > requirements.txt
railway up
```

### "Model not found"
- Train locally first: `python src/train.py`
- Commit the model: `git add model/`
- Push to GitHub
- Redeploy: `railway up`

---

## 📌 Important Notes

1. **Model Size**: The trained model file is stored in `model/` directory
2. **Training Data**: Person folders are in `persons/` directory
3. **Free Tier**: Railway free tier includes ~$5 credit monthly
4. **Auto-deploy**: Enable "Redeploy on push" for automatic updates on GitHub commits
5. **Environment Variables**: Add via Railway dashboard if needed

---

## 🎯 Complete Workflow

```bash
# 1. Train locally (if needed)
python src/train.py

# 2. Commit all changes
git add .
git commit -m "Ready for Railway deployment"
git push

# 3. Deploy to Railway
railway up

# 4. Copy the URL from Railway output

# 5. Update src/client.html with the URL

# 6. Deploy frontend separately (if needed)

# 7. Test at https://your-app.railway.app
```

---

## 💰 Pricing

- **Free Tier**: $5/month credit (usually covers small apps)
- **Pay as you go**: After free credit expires, only pay for what you use
- **No credit card required** initially on free tier

---

## 📚 Resources

- Railway Docs: https://docs.railway.app
- FastAPI Docs: https://fastapi.tiangolo.com
- API Endpoints: See `API_DOCUMENTATION.md`

---

**Status**: ✅ Ready to deploy!

**Next Command**: `railway login` then `railway up`
