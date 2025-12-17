# GitHub Repository Setup for PyMTCNN

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Settings:
   - Name: `PyMTCNN`
   - Description: `High-performance MTCNN face detection optimized for Apple Neural Engine - 34.26 FPS`
   - Public
   - Do NOT initialize with README, .gitignore, or license

3. Click "Create repository"

## Step 2: Initialize Git and Push

```bash
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3/pymtcnn

# Initialize git repo
git init
git add .
git commit -m "Initial commit - PyMTCNN v1.0.0"

# Add remote (replace YOUR-USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/PyMTCNN.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Update Package Metadata with Correct URLs

After creating the repo, update these files with your actual GitHub username:

### Update `setup.py`:
```python
url="https://github.com/YOUR-USERNAME/PyMTCNN",
```

### Update `pyproject.toml`:
```toml
[project.urls]
Homepage = "https://github.com/YOUR-USERNAME/PyMTCNN"
Documentation = "https://github.com/YOUR-USERNAME/PyMTCNN/blob/main/README.md"
Repository = "https://github.com/YOUR-USERNAME/PyMTCNN"
"Bug Tracker" = "https://github.com/YOUR-USERNAME/PyMTCNN/issues"
```

Then commit and push the changes:
```bash
git add setup.py pyproject.toml
git commit -m "Update GitHub URLs"
git push
```

## Step 4: Link PyPI to GitHub (Optional but Recommended)

### Option A: Manual Setup (Simple)

Your PyPI package will automatically link to GitHub if the URLs in `setup.py` and `pyproject.toml` are correct. When you upload to PyPI, it will show:
- Homepage link → Your GitHub repo
- Source code link → Your GitHub repo
- Bug tracker link → GitHub issues

**That's it!** The metadata in your package does the linking.

### Option B: GitHub Actions for Automated Publishing (Advanced)

If you want to automatically publish to PyPI when you create a GitHub release:

1. **Add PyPI API token to GitHub Secrets**:
   - Go to your PyMTCNN repo on GitHub
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token (starts with `pypi-...`)
   - Click "Add secret"

2. **Create GitHub Actions workflow**:

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

3. **How it works**:
   - Create a GitHub release (e.g., v1.0.1)
   - GitHub Actions automatically builds and uploads to PyPI
   - No manual `twine upload` needed!

### Option C: Link Existing PyPI Package to GitHub

If you've already published to PyPI:

1. Log in to https://pypi.org
2. Go to your project: https://pypi.org/project/pymtcnn/
3. Click "Manage" → "Settings"
4. The links should already be there from your package metadata
5. If not, you can rebuild and re-upload with updated metadata

## Step 5: Create First Release

After pushing to GitHub:

1. Go to your repo: https://github.com/YOUR-USERNAME/PyMTCNN
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0.0`
4. Title: `PyMTCNN v1.0.0 - High-Performance Face Detection`
5. Description:
```markdown
# PyMTCNN v1.0.0

High-performance MTCNN face detection optimized for Apple Neural Engine.

## Installation
```bash
pip install pymtcnn
```

## Performance
- 34.26 FPS with batch processing
- 175.7x speedup over baseline
- 95% IoU accuracy

## Links
- PyPI: https://pypi.org/project/pymtcnn/
- Documentation: [README.md](README.md)
```

6. Attach files (optional):
   - `dist/pymtcnn-1.0.0-py3-none-any.whl`
   - `dist/pymtcnn-1.0.0.tar.gz`

7. Click "Publish release"

## What You Get

After setup, your PyPI page will show:
- ✅ Homepage link to GitHub
- ✅ Source code link to GitHub
- ✅ Bug tracker link to GitHub Issues
- ✅ Automatic README rendering from PyPI
- ✅ Download badges
- ✅ License information

## Quick Summary

**For basic linking** (recommended):
1. Create GitHub repo named `PyMTCNN`
2. Push pymtcnn code to it
3. Update `setup.py` and `pyproject.toml` with correct GitHub URLs
4. Upload to PyPI with `twine upload dist/*`
5. PyPI automatically links to GitHub!

**For automated publishing** (optional):
- Add GitHub Actions workflow
- Add PyPI token to GitHub Secrets
- Create releases on GitHub → auto-publishes to PyPI
