#!/bin/bash
# Initialize Git repository and prepare for GitHub upload

echo "Initializing Git repository..."

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Music Theory AI Agent

- Multi-agent architecture for music theory analysis
- 20+ specialized music analysis tools
- Support for Humdrum/Kern notation format
- Multi-model support (OpenAI, Claude, Gemini)
- Comprehensive testing framework
- Documentation and examples"

echo "Git repository initialized successfully!"
echo ""
echo "To upload to GitHub:"
echo "1. Create a new repository on GitHub"
echo "2. Run: git remote add origin https://github.com/yourusername/music-theory-ai-agent.git"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
echo ""
echo "Repository is ready for GitHub upload!"
