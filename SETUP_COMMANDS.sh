#!/bin/bash
# Setup commands for PyMTCNN GitHub repository
#
# BEFORE RUNNING THIS SCRIPT:
# 1. Create GitHub repo at https://github.com/new named "PyMTCNN"
# 2. Replace YOUR-USERNAME below with your actual GitHub username
# 3. Make executable: chmod +x SETUP_COMMANDS.sh
# 4. Run: ./SETUP_COMMANDS.sh

# Set your GitHub username here
GITHUB_USERNAME="YOUR-USERNAME"  # ← CHANGE THIS!

echo "=========================================="
echo "PyMTCNN GitHub Setup"
echo "=========================================="
echo ""

# Navigate to pymtcnn directory
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3/pymtcnn

# Check if git is already initialized
if [ -d .git ]; then
    echo "Git repository already initialized"
else
    echo "Initializing git repository..."
    git init
fi

# Check if remote exists
if git remote | grep -q origin; then
    echo "Remote 'origin' already exists"
    git remote -v
else
    echo "Adding remote origin..."
    git remote add origin "https://github.com/${GITHUB_USERNAME}/PyMTCNN.git"
fi

# Stage all files
echo "Staging files..."
git add .

# Create initial commit if needed
if git rev-parse HEAD >/dev/null 2>&1; then
    echo "Repository already has commits"
else
    echo "Creating initial commit..."
    git commit -m "Initial commit - PyMTCNN v1.0.0

Production-ready MTCNN face detection package:
- 34.26 FPS with cross-frame batching
- 175.7x speedup over baseline
- 95% IoU accuracy
- Ready for PyPI publication

Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
fi

# Set main branch
echo "Setting main branch..."
git branch -M main

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "Repository: https://github.com/${GITHUB_USERNAME}/PyMTCNN"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Successfully pushed to GitHub!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. View your repo: https://github.com/${GITHUB_USERNAME}/PyMTCNN"
    echo "2. Create a release (v1.0.0)"
    echo "3. Upload to PyPI with: twine upload dist/*"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Push failed"
    echo "=========================================="
    echo ""
    echo "Common issues:"
    echo "1. Check if you created the GitHub repo"
    echo "2. Verify your GitHub username is correct"
    echo "3. Make sure you have push permissions"
    echo ""
fi
