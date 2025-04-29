# Drinking Water Quality Prediction

A machine learning system that predicts drinking water quality based on chemical and physical parameters. The system uses multiple ML models to analyze water quality indicators and determine if water is safe for consumption.

## Features

- Multiple ML models (Random Forest, XGBoost, LightGBM, Decision Tree, Linear SVC)
- Interactive web interface for predictions
- Real-time model training and evaluation
- Comprehensive data visualization
- Robust data validation and preprocessing
- Automated testing and CI/CD pipeline
- Comprehensive documentation

## Project Structure

```
drinking-water-quality/
├── data/               # Data files and processing
├── docs/              # Documentation files
├── models/            # Trained model files
├── notebooks/         # Jupyter notebooks for analysis
├── reports/           # Analysis and evaluation reports
├── scripts/           # Utility scripts
├── src/               # Source code
├── tests/             # Test files
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
├── .github/           # GitHub workflows and templates
├── config.yaml        # Configuration file
├── pytest.ini         # PyTest configuration
├── pyproject.toml     # Project metadata and dependencies
├── requirements.txt   # Python dependencies
├── run_app.py         # Main application entry point
└── README.md          # Project documentation
```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd drinking-water-quality
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the web application:

   ```bash
   python run_app.py
   ```

2. Open your web browser and navigate to:

   ```
   http://localhost:8501
   ```

3. Use the navigation menu to:
   - View data overview and statistics
   - Train and evaluate models
   - Make water quality predictions

## Data Format

The system expects water quality data with the following parameters:

- pH (0-14)
- Iron (mg/L)
- Nitrate (mg/L)
- Chloride (mg/L)
- Lead (mg/L)
- Zinc (mg/L)
- Turbidity (NTU)
- Fluoride (mg/L)
- Copper (mg/L)
- Sulfate (mg/L)
- Conductivity (µS/cm)
- Chlorine (mg/L)
- Total Dissolved Solids (mg/L)
- Water Temperature (°C)
- Air Temperature (°C)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black .

# Check linting
flake8 .

# Check types
mypy .
```

### Model Training

1. Navigate to the "Model Training" page in the web interface
2. Click "Train Models" to train all models with default parameters
3. View performance metrics for each model
4. Models are automatically saved for future use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details
