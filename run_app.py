import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

# Run the Streamlit app
os.system(f"streamlit run {project_root}/src/app.py")
