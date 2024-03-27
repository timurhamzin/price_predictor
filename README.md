# Price Predictor

## Project Overview
This project is a Price Predictor designed as a Minimum Viable Product (MVP) to showcase the flow of user interaction and capabilities of predicting prices based on various features. It uses Streamlit for the user interface, allowing users to load data, select features, train a model, and make predictions.

### Features
- Data loading and cleaning from CSV files
- Feature selection based on correlation analysis
- Model training with options for Random Forest Regressor and Linear Regression
- Interactive price prediction based on user inputs

## Getting Started

### Prerequisites
- Python 3.9 or later
- pip for installing dependencies

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd path/to/price_predictor
   ```
3. Install required Python packages in an activated virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
- Prepare your dataset in CSV format.
- Ensure it includes the necessary features for training and prediction.
- Set the target feature in the `config.py` file.

## Usage

### Running the Application
1. Start the Streamlit application:
   ```bash
   streamlit run main.py
   ```
2. Open your web browser and go to the address shown in the terminal (usually `http://localhost:8501`).

### Steps to Follow
1. **Data Loading**: Upload your CSV file and select columns to include.
2. **Feature Selection**: Choose features based on correlation analysis.
3. **Model Training**: Select a model and configure its parameters.
4. **Prediction**: Enter values for the selected features to predict prices.

## Project Structure
```
.
└── price_predictor
    ├── loading_and_selection.py  # Handles data loading and feature selection
    ├── main.py                    # Main application script for Streamlit
    ├── model_training_and_eval.py # Handles model training and evaluation
    ├── prediction.py              # Manages prediction logic and user input collection for prediction
    └── README.md                  # Project documentation
```

### Key Files and Their Roles
- `main.py`: The entry point of the application. It integrates different modules and Streamlit UI components.
- `loading_and_selection.py`: Contains functions for loading data, cleaning, and selecting features based on user input.
- `model_training_and_eval.py`: Includes functions for training machine learning models and evaluating their performance.
- `prediction.py`: Handles the collection of user inputs for features and uses the trained model to predict prices.

## Further Development
This project is designed as an MVP. Future enhancements could include advanced model tuning, the integration of more sophisticated feature selection techniques, and deployment strategies for scaling the application.
