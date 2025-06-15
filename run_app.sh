#!/bin/bash

# CohortManager Launch Script
echo "🏥 Starting CohortManager..."
echo "This will open the app in your default browser at http://localhost:8501"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Launch the app
echo "🚀 Launching CohortManager..."
streamlit run app.py 