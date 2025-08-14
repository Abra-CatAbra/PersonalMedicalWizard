# Project Context for AI Assistant

## Project Overview
PersonalMedicalWizard - AI-powered chest X-ray analysis tool using DenseNet-121 for detecting 18 pathologies.

## Current Status
- ✅ Basic MVP working with FastAPI backend and HTML frontend
- ✅ DenseNet model integration via torchxrayvision
- 🚧 Needs optimization and production-ready features

## Priority Phases

### Phase 1: Stability (Current)
- Add comprehensive error handling
- Input validation for image uploads
- Graceful failure modes
- Basic logging

### Phase 2: Performance
- Implement Redis caching
- Add async processing
- Optimize model loading
- Client-side image validation

### Phase 3: Features
- DICOM support
- PDF report generation
- Batch processing
- Historical tracking

## Architecture Decisions
- FastAPI for async Python backend
- Keep frontend simple (vanilla JS) until core is stable
- Use Redis for caching when added
- Models stored locally, not in git

## Testing Commands
```bash
# Backend
cd chest-xray-analyzer/backend
source venv/bin/activate
python -m pytest tests/

# Linting
python -m black .
python -m flake8 .
```

## Development Guidelines
1. Always test changes with actual X-ray images
2. Maintain medical disclaimer prominently
3. Never commit model files or patient data
4. Focus on one phase at a time
5. Create small, focused commits

## Known Issues
- Model loads on every startup (needs lazy loading)
- No input validation on file types
- Missing error handling for corrupted images
- No request rate limiting

## Next Immediate Tasks
1. Add try-catch blocks to main.py endpoints
2. Validate image format before processing
3. Add request logging
4. Create basic test suite

## File Structure
```
PersonalMedicalWizard/
├── chest-xray-analyzer/
│   ├── backend/          # FastAPI application
│   ├── frontend/         # Web interface
│   ├── models/           # Model weights (gitignored)
│   └── tests/            # Test suite
```

## Important Notes
- This is a medical prototype - NOT for clinical use
- All processing happens locally for privacy
- Model accuracy depends on image quality
- Always show confidence scores with results