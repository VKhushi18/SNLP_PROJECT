#!/bin/bash

# CleanSpeak Setup Script
echo "💬 Setting up CleanSpeak: AI Toxic Comment Detector"
echo "=================================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   source venv/bin/activate"
echo "   streamlit run app.py"
echo ""
echo "🎉 Enjoy using CleanSpeak!"

