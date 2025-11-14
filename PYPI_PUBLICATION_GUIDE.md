# PyPI Publication Guide for PyMTCNN v1.0.0

This guide walks through publishing PyMTCNN to the Python Package Index (PyPI).

## Prerequisites

- [x] Package built successfully (`dist/pymtcnn-1.0.0-py3-none-any.whl` & `dist/pymtcnn-1.0.0.tar.gz`)
- [x] Distributions validated with `twine check` (PASSED)
- [ ] PyPI account created
- [ ] TestPyPI account created (recommended for testing)

## Distribution Files

Built distributions are located in `dist/`:
- **Wheel**: `pymtcnn-1.0.0-py3-none-any.whl` (929KB)
- **Source**: `pymtcnn-1.0.0.tar.gz` (939KB)

## Step 1: Create PyPI Accounts

### Create TestPyPI Account (Recommended First)
1. Go to https://test.pypi.org/account/register/
2. Create an account with email and password
3. Verify your email address
4. Enable 2FA (required for uploading packages)

### Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Create an account with email and password
3. Verify your email address
4. Enable 2FA (required for uploading packages)

## Step 2: Create API Tokens

### TestPyPI API Token
1. Log in to https://test.pypi.org
2. Go to Account Settings → API tokens
3. Click "Add API Token"
4. Name: `pymtcnn-upload`
5. Scope: "Entire account" (or specific project after first upload)
6. **SAVE THE TOKEN** - you can only see it once!

### PyPI API Token
1. Log in to https://pypi.org
2. Go to Account Settings → API tokens
3. Click "Add API Token"
4. Name: `pymtcnn-upload`
5. Scope: "Entire account" (or specific project after first upload)
6. **SAVE THE TOKEN** - you can only see it once!

## Step 3: Configure Credentials

### Option A: Use .pypirc file (Recommended)

Create/edit `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

**Security**:
```bash
chmod 600 ~/.pypirc  # Restrict permissions
```

### Option B: Use Environment Variables

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE
```

## Step 4: Test Upload to TestPyPI (Recommended)

```bash
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3/pymtcnn

# Upload to TestPyPI
/Users/johnwilsoniv/Library/Python/3.10/bin/twine upload --repository testpypi dist/*
```

### Test Installation from TestPyPI

```bash
# Create test environment
python3 -m venv test_env
source test_env/bin/activate

# Install from TestPyPI (note: dependencies will come from real PyPI)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pymtcnn

# Test import
python3 -c "from pymtcnn import CoreMLMTCNN; print('Import successful!')"

# Deactivate and remove test environment
deactivate
rm -rf test_env
```

## Step 5: Upload to PyPI (Production)

**IMPORTANT**: Once uploaded to PyPI, you CANNOT delete or re-upload the same version. Make sure everything is correct!

```bash
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3/pymtcnn

# Upload to PyPI
/Users/johnwilsoniv/Library/Python/3.10/bin/twine upload dist/*
```

You'll see output like:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading pymtcnn-1.0.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 929.0/929.0 kB • 00:01 • ?
Uploading pymtcnn-1.0.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 939.0/939.0 kB • 00:01 • ?

View at:
https://pypi.org/project/pymtcnn/1.0.0/
```

## Step 6: Verify Publication

### Check PyPI Web Interface
1. Visit https://pypi.org/project/pymtcnn/
2. Verify package information, description, and links
3. Check that README renders correctly
4. Verify license and classifiers

### Test Installation
```bash
# Create clean environment
python3 -m venv verify_env
source verify_env/bin/activate

# Install from PyPI
pip install pymtcnn

# Test import and basic functionality
python3 -c "
from pymtcnn import CoreMLMTCNN
print('PyMTCNN v1.0.0 installed successfully!')
detector = CoreMLMTCNN()
print('Detector initialized!')
"

# Clean up
deactivate
rm -rf verify_env
```

## Step 7: Update GitHub Repository

### Tag the Release
```bash
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3

# Create git tag
git tag -a v1.0.0 -m "Release v1.0.0 - PyMTCNN production-ready
- 34.26 FPS with cross-frame batching
- 175.7x speedup over baseline
- Published to PyPI"

# Push tag to GitHub
git push origin v1.0.0
```

### Create GitHub Release
1. Go to https://github.com/your-org/PyMTCNN/releases
2. Click "Create a new release"
3. Select tag: `v1.0.0`
4. Release title: `PyMTCNN v1.0.0 - High-Performance Face Detection`
5. Description:
```markdown
# PyMTCNN v1.0.0

High-performance MTCNN face detection optimized for Apple Neural Engine.

## Installation

```bash
pip install pymtcnn
```

## Performance
- **34.26 FPS** with batch processing
- **175.7x speedup** over baseline Python
- **95% IoU accuracy** vs C++ OpenFace

## Quick Start
```python
from pymtcnn import CoreMLMTCNN

detector = CoreMLMTCNN()
bboxes, landmarks = detector.detect(image)
```

## Links
- PyPI: https://pypi.org/project/pymtcnn/
- Documentation: [README.md](README.md)
- Examples: [examples/](examples/)

## What's New
- Initial release
- Production-ready package for S1 integration
- Comprehensive documentation and examples
```

6. Attach distribution files (optional):
   - `pymtcnn-1.0.0-py3-none-any.whl`
   - `pymtcnn-1.0.0.tar.gz`

7. Click "Publish release"

## Step 8: Announce Release

### Update README Badges (Optional)
Add badges to `README.md`:
```markdown
[![PyPI version](https://badge.fury.io/py/pymtcnn.svg)](https://badge.fury.io/py/pymtcnn)
[![Downloads](https://pepy.tech/badge/pymtcnn)](https://pepy.tech/project/pymtcnn)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
```

## Troubleshooting

### "HTTPError: 403 Forbidden"
- Check that your API token is correct
- Verify 2FA is enabled on your account
- Use `__token__` as username (not your PyPI username)

### "File already exists"
- PyPI doesn't allow re-uploading the same version
- Increment version number (e.g., 1.0.0 → 1.0.1)
- Rebuild distributions: `python3 -m build`

### "Invalid distribution"
- Run `twine check dist/*` to validate
- Fix any errors in metadata
- Rebuild if necessary

### Large File Size Warning
- PyMTCNN wheel is ~929KB (within PyPI limits)
- Source tarball is ~939KB (within PyPI limits)
- No action needed

## Future Updates

### To Publish Version 1.0.1 (or later)

1. **Update version** in:
   - `pymtcnn/__init__.py` (`__version__ = "1.0.1"`)
   - `pyproject.toml` (`version = "1.0.1"`)
   - `setup.py` (`version="1.0.1"`)

2. **Commit changes**:
   ```bash
   git add .
   git commit -m "Bump version to 1.0.1"
   ```

3. **Rebuild distributions**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   python3 -m build
   twine check dist/*
   ```

4. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

5. **Create git tag**:
   ```bash
   git tag -a v1.0.1 -m "Release v1.0.1"
   git push origin v1.0.1
   ```

## Support

- **Issues**: https://github.com/your-org/PyMTCNN/issues
- **PyPI**: https://pypi.org/project/pymtcnn/
- **License**: CC BY-NC 4.0

---

**Status**: ✅ Ready for PyPI publication
**Version**: 1.0.0
**Built**: November 14, 2025
