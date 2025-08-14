#!/bin/bash

echo "🏥 Setting up Chest X-Ray Analyzer..."
echo "=================================="

cd backend

echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "1. Start the backend:"
echo "   cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo ""
echo "2. Open the frontend:"
echo "   Open frontend/index.html in your browser"
echo ""
echo "The API will be available at: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"