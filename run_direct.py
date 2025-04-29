import os
import sys
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).parent.absolute()

# Add the src directory to Python path
sys.path.insert(0, str(project_root))

# Change to the src directory
os.chdir(project_root / "src")

# Import and run the app
import app
app.main()
