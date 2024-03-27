import pandas as pd
import streamlit as st

from config import target_feature


def handle_prediction():
    st.subheader('The values you enter must fall within the range of values used for training.')

    if 'trained_final_model' not in st.session_state or 'preprocessor' not in st.session_state:
        st.error("No trained model available. Please train the model first.")
        return

    data_for_training = st.session_state['data_cleaned']
    selected_features = st.session_state.get('original_features_selected_for_modeling', [])

    # Collect user input for each feature
    user_input_features = {}
    for feature in selected_features:
        if "latitude" in feature or "longitude" in feature:
            # For latitude and longitude, use float inputs without converting to int
            value = data_for_training[feature].dropna().median()
            min_value = data_for_training[feature].min()
            max_value = data_for_training[feature].max()
            step = 0.00001
            user_input_features[feature] = st.number_input(f"Input {feature}", value=float(value),
                                                           min_value=float(min_value), max_value=float(max_value),
                                                           step=step)
        elif pd.api.types.is_numeric_dtype(data_for_training[feature]):
            # Calculate median of non-null values
            median_value = data_for_training[feature].dropna().median()
            if pd.notnull(median_value):
                value = int(median_value) if not ("latitude" in feature or "longitude" in feature) else float(
                    median_value)
            else:
                value = 0  # Default value in case all values are null

            min_value = int(data_for_training[feature].min()) if pd.notnull(data_for_training[feature].min()) else value
            max_value = int(data_for_training[feature].max()) if pd.notnull(data_for_training[feature].max()) else value
            user_input_features[feature] = st.number_input(f"Input {feature}", value=value, min_value=min_value,
                                                           max_value=max_value, step=1)
        else:
            # Categorical feature handling
            options = data_for_training[feature].dropna().unique()
            default_index = 0
            user_input_features[feature] = st.selectbox(f"Select {feature}", options=options, index=default_index)

    # Prepare the data for prediction
    input_data = pd.DataFrame([user_input_features])

    # Explicitly convert categorical features to 'object' type to match training data
    for feature in selected_features:
        if not pd.api.types.is_numeric_dtype(data_for_training[feature]):
            input_data[feature] = input_data[feature].astype('object')

    # Transform the input data using the saved preprocessor
    input_data_transformed = st.session_state['preprocessor'].transform(input_data)

    # Make prediction
    prediction = st.session_state['trained_final_model'].predict(input_data_transformed)

    # Display the prediction
    st.header(f"Predicted {target_feature}: {round(prediction[0], 2)}")
