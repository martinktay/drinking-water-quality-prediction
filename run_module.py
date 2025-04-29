import os
import sys
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).parent.absolute()

# Add the project root to Python path
sys.path.insert(0, str(project_root))

# Run the app as a module
os.system(f"python -m streamlit run src/app.py")
