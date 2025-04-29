import os
import sys
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).parent.absolute()

# Add the project root to Python path
sys.path.insert(0, str(project_root))

# Change to the src directory
os.chdir(project_root / "src")

# Run the Streamlit app
os.system("streamlit run app.py")
