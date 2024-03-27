import streamlit as st
from loading_and_selection import handle_data_loading, initialize_session_state, handle_feature_selection
from model_training_and_eval import create_placeholders, handle_model_training
from prediction import handle_prediction

def main() -> None:
    """
    Main application script for the Real Estate Price Predictor. It initializes the session state,
    sets the sidebar title, and manages navigation between different pages of the application,
    including data loading, feature selection, model training, and prediction.
    """
    # Initialize session state
    initialize_session_state()

    # Set up the sidebar title and page navigation
    st.sidebar.title('Real Estate Price Predictor')
    page = st.sidebar.radio('Select a step', ('Data Loading', 'Feature Selection', 'Model Training', 'Prediction'))

    # Handle page navigation
    if page == 'Data Loading':
        handle_data_loading()
    elif page == 'Feature Selection':
        handle_feature_selection()
    elif page == 'Model Training':
        handle_model_training_page()
    elif page == 'Prediction':
        handle_prediction()

def handle_model_training_page() -> None:
    """
    Handles the logic for the Model Training page. It checks if features are ready for modeling,
    creates placeholders for the UI, and calls the function to handle model training.
    """
    if st.session_state.get('features_ready_for_modeling'):
        placeholders = create_placeholders()
        handle_model_training(placeholders)
    else:
        st.warning("Please complete the feature selection and save the features for modeling before proceeding.")

if __name__ == '__main__':
    main()
