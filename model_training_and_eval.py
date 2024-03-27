from typing import Tuple, Iterable

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config import target_feature
from utils import display_dataframe_in_placeholder


def prepare_features_overview(data_for_training, st):
    """
    Prepare a DataFrame describing features for the Streamlit app, excluding features
    discarded due to high cross-correlation or low correlation with the target.
    """
    features_info = []
    # Combine lists of features to exclude from both high cross-correlation and low target correlation
    features_to_exclude = set(
        st.session_state.get('features_to_exclude_high_cross_corr', [])
    ).union(st.session_state.get('low_target_corr_features', []))

    for column in data_for_training.columns:
        # Exclude specific features based on combined list
        if column in features_to_exclude:
            continue
        feature_dict = {'Feature': column, 'Type': data_for_training[column].dtype}
        # Check for both 'int64' and 'int32' for numerical features
        if data_for_training[column].dtype in ['float64', 'int64', 'int32']:
            feature_dict.update({
                'Category': 'Numerical',
                'Min': data_for_training[column].min(),
                'Max': data_for_training[column].max(),
                'Mean': data_for_training[column].mean(),
                'Median': data_for_training[column].median(),
                'Std': data_for_training[column].std(),
            })
        else:
            feature_dict.update({
                'Category': 'Categorical',
                'Unique Values': data_for_training[column].nunique(),
                'Top Value': data_for_training[column].mode().iloc[0] if not data_for_training[
                    column].mode().empty else 'N/A',
            })
        features_info.append(feature_dict)
    return pd.DataFrame(features_info)


def configure_model(model_choice, placeholders):
    """
    Configure and return the selected machine learning model using individual placeholders for each setting.
    """
    if model_choice == 'Random Forest Regressor':
        n_estimators = placeholders['n_estimators'].number_input(
            'Number of trees in the forest',
            min_value=10, max_value=1000, value=100, step=10,
            key=f'n_estimators_{model_choice}')  # Unique key
        max_depth = placeholders['max_depth'].number_input(
            'Maximum depth of the tree (None for unlimited)',
            min_value=1, max_value=None, value=10, step=1,
            key=f'max_depth_{model_choice}')  # Unique key
        random_state = placeholders['random_state'].number_input(
            'Random state', value=42,
            key=f'random_state_{model_choice}')  # Unique key
        n_jobs = placeholders['n_jobs'].number_input(
            'Number of jobs to run in parallel', min_value=-1, value=-1,
            key=f'n_jobs_{model_choice}')  # Unique key
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
                                      n_jobs=n_jobs)
    elif model_choice == 'Linear Regression':
        fit_intercept = placeholders['fit_intercept'].checkbox(
            'Fit Intercept', value=True,
            key=f'fit_intercept_{model_choice}')  # Unique key
        model = LinearRegression(fit_intercept=fit_intercept)
    else:
        raise ValueError("Invalid model choice")
    return model


def evaluate_model(preprocessor, model, X_test, y_test):
    """
    Evaluate the model and return evaluation metrics.
    Apply the preprocessor transformations learned from the training data to the test data.
    """
    # Correctly transform the test data using the preprocessor learned from the training data
    X_test_preprocessed = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_preprocessed)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2


def get_feature_importances(pipeline, preprocessor):
    """
    Extract and return feature importances if the model supports it.
    """
    # Extract the regressor from the pipeline
    regressor = pipeline.named_steps['regressor']

    if hasattr(regressor, 'feature_importances_'):
        # Extract feature names from the preprocessor after fitting
        feature_names_out = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_

        assert len(feature_names_out) == len(importances), (
            "Feature names and importances must have the same length.")
        importances_df = pd.DataFrame({
            'Feature': feature_names_out,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return importances_df
    else:
        return pd.DataFrame({'Feature': [
            'Feature importances not available for this model type'], 'Importance': [0]})


def prepare_data_for_training(data_for_training, test_size=0.2) -> Tuple[
    Iterable[str], Iterable[str], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target = data_for_training[target_feature]
    features = data_for_training.drop(columns=[target_feature])
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42)
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns
    return numerical_features, categorical_features, X_train, X_test, y_train, y_test


def calculate_preliminary_feature_importances(X_train, y_train, numerical_features, categorical_features):
    """
    Calculate preliminary feature importances using a RandomForestRegressor.

    Parameters:
    - X_train: The preprocessed training data.
    - y_train: The target variable for the training data.
    - numerical_features: List of names of numerical features.
    - categorical_features: List of names of categorical features.

    Returns:
    - DataFrame with features and their importances.
    """
    # Setup the ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    # Fit the preprocessor to the training data and transform X_train
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    if X_train_preprocessed.shape[1] == 0:  # Check if no features are available after preprocessing
        st.error(
            ("No features available for training after preprocessing. "
             "Please review your feature selection and preprocessing steps."))
        return

    # Use a RandomForestRegressor as the preliminary model
    preliminary_model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1)

    # Train the preliminary model on preprocessed data
    preliminary_model.fit(X_train_preprocessed, y_train)

    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Extract feature importances
    importances = preliminary_model.feature_importances_

    # Create DataFrame with features and their importances
    features_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return features_importances_df


def train_model(model, X_train, y_train, numerical_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)

    if X_train_preprocessed.shape[1] == 0:
        st.error(
            ("No features available for training after preprocessing. "
             "Please review your feature selection and preprocessing steps."))
        return

    model.fit(X_train_preprocessed, y_train)

    return preprocessor, model


def handle_feature_importances(training_data, training_targets, numerical_features_list, categorical_features_list,
                               ui_placeholders):
    # Calculate preliminary feature importances based on training data
    preliminary_feature_importances = calculate_preliminary_feature_importances(
        training_data, training_targets, numerical_features_list, categorical_features_list)

    if preliminary_feature_importances is None:
        return

    # Initially mark the top 20 features as important for modeling
    number_of_initial_features = 20
    preliminary_feature_importances[
        'Use for Modeling'] = preliminary_feature_importances.index < number_of_initial_features

    # Cache the initial feature importances in the session state
    st.session_state['cached_feature_importances'] = preliminary_feature_importances

    # Copy the cached feature importances to modify and display
    displayed_feature_importances = st.session_state['cached_feature_importances'].copy()

    # Display the title for the feature importances section
    ui_placeholders['feature_importance_title'].write("### Preliminary Feature Importances")

    # Display the dataframe of feature importances
    display_dataframe_in_placeholder(ui_placeholders['feature_importances_display'],
                                     displayed_feature_importances[['Feature', 'Importance', 'Use for Modeling']])

    # Generate options for the number of features the user can select for modeling
    feature_selection_options = list(range(1, len(displayed_feature_importances) + 1))

    # Allow the user to select the number of most important features for modeling
    selected_features_count = ui_placeholders['feature_count_selector'].selectbox(
        'Select the number of the most important features to use', feature_selection_options,
        index=min(len(feature_selection_options) - 1, number_of_initial_features - 1))

    # Reset all features to 'not used for modeling' before updating based on user selection
    displayed_feature_importances['Use for Modeling'] = False
    displayed_feature_importances.loc[:selected_features_count - 1, 'Use for Modeling'] = True

    # Update the display with the new selection
    display_dataframe_in_placeholder(ui_placeholders['feature_importances_display'], displayed_feature_importances)

    # Update the session state with the new selection for future reference
    st.session_state['cached_feature_importances'] = displayed_feature_importances

    # Return the updated feature importances dataframe
    return displayed_feature_importances


def create_placeholders():
    return {
        'features_overview': {
            'title': st.empty(),
            'display': st.empty(),
        },
        'feature_importance': {
            'button': st.empty(),
            'title': st.empty(),
            'display': st.empty(),
            'num_features_to_use': st.empty(),  # Placeholder for the selectbox
        },
        'model_config': {
            'selection': st.empty(),  # Placeholder for model selection
            'n_estimators': st.empty(),
            'max_depth': st.empty(),
            'random_state': st.empty(),
            'n_jobs': st.empty(),
            'fit_intercept': st.empty(),  # This one is for Linear Regression
        },
        'model_training': {
            'button': st.empty(),
        },
        'model_evaluation': {
            'title': st.empty(),
            'mae': st.empty(),
            'mse': st.empty(),
            'r2': st.empty(),
        },
    }


def report_evaluation_metrics(preprocessor, model, X_test, y_test, evaluation_placeholders):
    mae, mse, r2 = evaluate_model(preprocessor, model, X_test, y_test)
    evaluation_placeholders['title'].subheader("Evaluation Metrics")
    evaluation_placeholders['mae'].metric("Mean Absolute Error", f"{mae:.2f}")
    evaluation_placeholders['mse'].metric("Mean Squared Error", f"{mse:.2f}")
    evaluation_placeholders['r2'].metric("R^2 Score", f"{r2:.2f}")


def train_and_evaluate_model(selected_features_df, data_for_training, target_feature, untrained_final_model,
                             placeholders):
    if selected_features_df is None:
        return

    # Ensure there are selected features for modeling
    if selected_features_df[selected_features_df['Use for Modeling']].empty:
        st.error("No features selected for modeling. Please select features before training the model.")
        return

    selected_features = selected_features_df[selected_features_df['Use for Modeling']]['Feature'].tolist()
    if not selected_features:
        st.error('No features have been selected for modeling. Please go back and select features.')
        return

    st.session_state['original_features_selected_for_modeling'] = [
        feat for feat in data_for_training.columns if
        any(feat in selected for selected in selected_features)]

    # Check if there are any features to model on after selection
    if not st.session_state['original_features_selected_for_modeling']:
        st.error("Error: No features available for modeling. Please ensure that features are selected.")
        return

    numerical_features, categorical_features, X_train, X_test, y_train, y_test = prepare_data_for_training(
        data_for_training[st.session_state['original_features_selected_for_modeling'] + [target_feature]])

    if X_train.empty:
        st.error('Training data is empty after feature selection. Please adjust your feature selection criteria.')
        return

    st.session_state['preprocessor'], st.session_state['trained_final_model'] = train_model(
        untrained_final_model, X_train, y_train, numerical_features, categorical_features)

    report_evaluation_metrics(
        st.session_state['preprocessor'], st.session_state['trained_final_model'], X_test, y_test,
        placeholders['model_evaluation']
    )


def handle_model_training(placeholders):
    if 'data_cleaned' in st.session_state and st.session_state['data_cleaned'] is not None:
        data_for_training = st.session_state['data_cleaned'].copy()
        features_dataframe = prepare_features_overview(data_for_training, st)
        numerical_features, categorical_features, X_train, X_test, y_train, y_test = prepare_data_for_training(
            data_for_training)
        placeholders['features_overview']['title'].write("### Features Overview")
        display_dataframe_in_placeholder(placeholders['features_overview']['display'], features_dataframe)

        # Preliminary step for feature importance
        if placeholders['feature_importance']['button'].button(
                'Select the Most Important Features') or 'feature_importances_cached' in st.session_state:
            importances_df = handle_feature_importances(
                X_train, y_train, numerical_features, categorical_features, {
                    'feature_importance_title': placeholders['feature_importance']['title'],
                    'feature_importances_display': placeholders['feature_importance']['display'],
                    'feature_count_selector': placeholders['feature_importance']['num_features_to_use'],
                    'model_selection': placeholders['model_config']['selection'],
                })

            # Use placeholders to select a model
            model_choice = placeholders['model_config']['selection'].selectbox(
                'Select a model', ['Random Forest Regressor', 'Linear Regression'], key='model_choice')
            untrained_final_model = configure_model(model_choice, {
                'n_estimators': placeholders['model_config']['n_estimators'],
                'max_depth': placeholders['model_config']['max_depth'],
                'random_state': placeholders['model_config']['random_state'],
                'n_jobs': placeholders['model_config']['n_jobs'],
                'fit_intercept': placeholders['model_config']['fit_intercept'],  # This is for Linear Regression
            })

            # Update cached dataframe with new selection
            st.session_state['feature_importances_cached'] = importances_df
            if placeholders['model_training']['button'].button('Train Model'):
                placeholders['model_evaluation']['title'].subheader('Wait for evaluation metrics to appear...')
                train_and_evaluate_model(importances_df, data_for_training, target_feature, untrained_final_model,
                                         placeholders)
    else:
        st.error("Please ensure data is loaded and features are selected before training the model.")
