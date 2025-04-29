import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.quality_report = {}

    def load_data(self):
        """Load the dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data from {self.data_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def check_null_values(self):
        """Check for null values in the dataset."""
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100

        self.quality_report['null_values'] = {
            'counts': null_counts.to_dict(),
            'percentages': null_percentages.to_dict()
        }

        logger.info("Null value check completed")
        return null_percentages

    def check_data_types(self):
        """Validate data types of each column."""
        expected_types = {
            'pH': 'float64',
            'Iron': 'float64',
            'Nitrate': 'float64',
            'Chloride': 'float64',
            'Lead': 'float64',
            'Zinc': 'float64',
            'Turbidity': 'float64',
            'Fluoride': 'float64',
            'Copper': 'float64',
            'Sulfate': 'float64',
            'Conductivity': 'float64',
            'Chlorine': 'float64',
            'Total Dissolved Solids': 'float64',
            'Water Temperature': 'float64',
            'Air Temperature': 'float64',
            'Target': 'int64'
        }

        type_mismatches = {}
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if actual_type != expected_type:
                    type_mismatches[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }

        self.quality_report['data_types'] = type_mismatches
        logger.info("Data type check completed")
        return type_mismatches

    def check_value_ranges(self):
        """Check if values are within expected ranges."""
        expected_ranges = {
            'pH': (0, 14),
            'Iron': (0, 10),
            'Nitrate': (0, 100),
            'Chloride': (0, 500),
            'Lead': (0, 0.05),
            'Zinc': (0, 5),
            'Turbidity': (0, 10),
            'Fluoride': (0, 2),
            'Copper': (0, 2),
            'Sulfate': (0, 500),
            'Conductivity': (0, 2000),
            'Chlorine': (0, 5),
            'Total Dissolved Solids': (0, 2000),
            'Water Temperature': (0, 40),
            'Air Temperature': (-10, 50),
            'Target': (0, 1)
        }

        range_violations = {}
        for col, (min_val, max_val) in expected_ranges.items():
            if col in self.df.columns:
                violations = self.df[
                    (self.df[col] < min_val) | (self.df[col] > max_val)
                ][col]
                if not violations.empty:
                    range_violations[col] = {
                        'count': len(violations),
                        'percentage': (len(violations) / len(self.df)) * 100
                    }

        self.quality_report['value_ranges'] = range_violations
        logger.info("Value range check completed")
        return range_violations

    def check_outliers(self, threshold=3):
        """Detect outliers using z-score method."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            if col != 'Target':  # Skip target variable
                z_scores = np.abs(
                    (self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_count = len(z_scores[z_scores > threshold])
                if outlier_count > 0:
                    outliers[col] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(self.df)) * 100
                    }

        self.quality_report['outliers'] = outliers
        logger.info("Outlier check completed")
        return outliers

    def generate_report(self):
        """Generate a comprehensive quality report."""
        if self.quality_report:
            report_path = Path('reports/data_quality_report.json')
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                import json
                json.dump(self.quality_report, f, indent=4)

            logger.info(f"Quality report saved to {report_path}")
            return self.quality_report
        return None


def main():
    data_path = 'data/processed/processed_data.csv'
    checker = DataQualityChecker(data_path)

    if checker.load_data():
        checker.check_null_values()
        checker.check_data_types()
        checker.check_value_ranges()
        checker.check_outliers()
        report = checker.generate_report()

        if report:
            logger.info("Data quality checks completed successfully")
        else:
            logger.error("Failed to generate quality report")


if __name__ == "__main__":
    main()
