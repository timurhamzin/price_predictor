from functools import partial

import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from config import target_feature
from utils import display_dataframe_in_placeholder


def initialize_session_state() -> None:
    """
    Initializes the Streamlit session state with default values for various keys. This setup is crucial for
    managing data cleaning, feature selection, and analysis settings throughout the application lifecycle.
    Default values are set for cleaned data, feature selection configurations, and features to exclude based
    on high cross-correlation or low correlation with the target variable.
    """
    default_state = {
        'data_cleaned': None,
        'feature_selection': {'target_feature': None, 'excluded_features': []},
        'features_to_exclude_high_cross_corr': set(),
        'low_target_corr_features': set(),
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def transform_date_features(data_frame: pd.DataFrame, reference_date: str = '1970-01-01') -> pd.DataFrame:
    """
    Transforms all date features within a DataFrame into several more granular columns (year, month, day, etc.),
    based on the distance from a reference date. This transformation aims to enhance model training by providing
    additional numeric and categorical features derived from date columns.

    Parameters:
    - data_frame: DataFrame containing the data with date columns to be transformed.
    - reference_date: The reference date used for calculating the number of days from each date feature.

    Returns:
    - A DataFrame with original date columns transformed into multiple granular features.
    """
    reference_date = pd.to_datetime(reference_date)
    for column in data_frame.columns:
        if np.issubdtype(data_frame[column].dtype, np.datetime64):
            data_frame = _transform_single_date_feature(data_frame, column, reference_date)
    return data_frame

def _transform_single_date_feature(data_frame: pd.DataFrame, column: str, reference_date: pd.Timestamp) -> pd.DataFrame:
    """
    Helper function to transform a single date feature within a DataFrame.

    Parameters:
    - data_frame: The DataFrame containing the date feature.
    - column: The name of the column containing date values to be transformed.
    - reference_date: The reference date used for daystamp calculation.

    Returns:
    - The DataFrame with the specified date feature transformed.
    """
    data_frame[f'{column}_year'] = data_frame[column].dt.year.astype('int64')
    data_frame[f'{column}_month'] = data_frame[column].dt.month.astype(str)
    data_frame[f'{column}_day'] = data_frame[column].dt.day.astype(str)
    data_frame[f'{column}_weekday'] = data_frame[column].dt.weekday.astype(str)
    data_frame[f'{column}_daystamp'] = (data_frame[column] - reference_date).dt.days.astype('int64')
    data_frame.drop(column, axis=1, inplace=True)
    return data_frame

def load_and_clean_data(data: pd.DataFrame, selected_columns: list[str], remove_outliers: bool = False,
                        fill_na_with_median: bool = True, export_top_n: int = None) -> pd.DataFrame:
    """
    Cleans the provided DataFrame by selecting specified columns, removing outliers, filling missing values,
    and exporting the top N rows based on the given parameters.

    Parameters:
    - data: The original DataFrame to be cleaned.
    - selected_columns: A list of column names to be included in the cleaned data.
    - remove_outliers: A boolean flag indicating whether to remove outliers based on the IQR method.
    - fill_na_with_median: A boolean flag indicating whether to fill missing numeric values with the median.
    - export_top_n: An integer specifying the number of top rows to be exported; if None, all rows are considered.

    Returns:
    - A cleaned DataFrame that has been processed according to the specified parameters.
    """
    if 'price' in selected_columns:
        selected_columns.remove('price')  # Remove "price" if present

    if target_feature not in selected_columns:
        selected_columns.append(target_feature)
    data_cleaned = data[selected_columns].copy()

    # Data type conversion logic
    # Convert 'date_sold' to datetime
    if 'date_sold' in data_cleaned.columns:
        data_cleaned['date_sold'] = pd.to_datetime(data_cleaned['date_sold'])
    data_cleaned = transform_date_features(data_cleaned)

    # Convert identifier features to 'object' (categorical)
    id_features = [feature for feature in
                   ['city_id', 'district_id', 'metro_station_id', 'series_id', 'wall_id', 'street_id']
                   if feature in data_cleaned.columns]
    data_cleaned[id_features] = data_cleaned[id_features].astype('object').replace('', np.nan)
    data_cleaned[id_features] = data_cleaned[id_features].fillna(-1).astype('object')

    # Adjust the mapping as necessary based on actual data values
    boolean_features = [feature for feature in ['closed_yard', 'two_levels'] if feature in data_cleaned.columns]
    for feature in boolean_features:
        data_cleaned[feature] = data_cleaned[feature].map(
            {'yes': True, 'no': False, 'True': True, 'False': False}).astype('boolean')

    # Correct 'area_balcony' if it's numerical represented as strings
    if 'area_balcony' in data_cleaned.columns:
        data_cleaned['area_balcony'] = pd.to_numeric(data_cleaned['area_balcony'],
                                                     errors='coerce')  # Converts to float, sets errors to NaN

    if 'territory' in data_cleaned.columns:
        # Split the 'territory' feature into individual options
        territory_options = data_cleaned['territory'].str.get_dummies(sep=',')
        # Prefix the column names to avoid any name collision with existing features
        territory_options = territory_options.add_prefix('territory_')
        territory_options = territory_options.replace(0, -1)
        # Concatenate the new binary columns with the original DataFrame
        data_cleaned = pd.concat([data_cleaned, territory_options], axis=1)
        # Optionally, drop the original 'territory' column if no longer needed
        data_cleaned.drop(columns=['territory'], inplace=True)

    numerical_features = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    data_cleaned[numerical_features] = data_cleaned[numerical_features].replace(0, np.nan)

    # Optionally fill NaN values with median for numerical features
    if fill_na_with_median:
        for feature in numerical_features:
            median_value = data_cleaned[feature].median()
            data_cleaned[feature].fillna(median_value, inplace=True)

    data_cleaned = data_cleaned.dropna(subset=[target_feature])

    # Optionally export top N rows of the cleaned data to a CSV file
    if export_top_n:
        data_cleaned = data_cleaned.head(export_top_n)

    if remove_outliers:
        for feature in numerical_features:
            Q1 = data_cleaned[feature].quantile(0.25)
            Q3 = data_cleaned[feature].quantile(0.75)
            IQR = Q3 - Q1
            data_cleaned = data_cleaned[
                ~((data_cleaned[feature] < (Q1 - 1.5 * IQR)) | (data_cleaned[feature] > (Q3 + 1.5 * IQR)))]


    data_cleaned.to_csv('data_cleaned.csv', index=False)
    st.write(f'Cleaned data to use for modeling has been exported to "data_cleaned.csv" for review.')

    return data_cleaned


def handle_data_loading():
    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        selected_columns = st.multiselect('Select columns', df.columns,
                                          default=(set(df.columns) - {'id', 'status', 'builder_id'}))
        remove_outliers = st.checkbox('Remove outliers', value=True)
        fill_na_with_median = st.checkbox(
            'Replace empty values by the median', value=True)
        export_top_n = st.number_input("Rows to use (exported to data_cleaned.csv), 0 means `all`", value=50000, min_value=0)
        if selected_columns:
            st.session_state['data_cleaned'] = load_and_clean_data(
                df, selected_columns, remove_outliers=remove_outliers, fill_na_with_median=fill_na_with_median, export_top_n=export_top_n)
            st.write(st.session_state['data_cleaned'])


def handle_feature_selection() -> None:
    """
    Manages the feature selection process based on user input through the Streamlit interface. It includes steps
    for high cross-correlation analysis, target correlation analysis, and the ability for users to manually select
    features for exclusion. Results affect the session state, influencing the data prepared for modeling.

    This function creates placeholders for dynamic UI elements, processes user input to update session state
    with selected features, and displays visualizations for feature correlation and importance. It ensures that
    the feature selection process is interactive and reflective of the user's choices.
    """

    if st.session_state['data_cleaned'] is not None:
        # Create placeholders for dynamic UI elements
        high_cross_corr_heatmap_placeholder = st.empty()
        low_target_corr_multiselect_placeholder = st.empty()
        low_target_corr_table_placeholder = st.empty()
        low_target_corr_charts_button_placeholder = st.empty()
        low_target_corr_charts_placeholder = st.empty()

        data_cleaned = st.session_state['data_cleaned']
        original_numerical_features, _ = select_features(data_cleaned)

        if target_feature in original_numerical_features:
            original_numerical_features.remove(target_feature)

        setup_high_cross_correlation_analysis(
            data_cleaned, original_numerical_features, high_cross_corr_heatmap_placeholder)
        setup_target_correlation_analysis(
            data_cleaned, original_numerical_features, low_target_corr_table_placeholder,
            low_target_corr_multiselect_placeholder)
        handle_save_features_and_data_for_modeling(data_cleaned, original_numerical_features)
        display_scatter_plots_for_discarded_features(
            st.session_state['data_cleaned'], target_feature, low_target_corr_charts_button_placeholder,
            low_target_corr_charts_placeholder)

    else:
        st.error('Please load and clean the data in the Data Loading step first.')


def get_feature_importance_from_corr_with_target(data):
    correlations_with_target = data.corrwith(data[target_feature]).abs()

    # Convert the series to a dictionary, with feature names as keys and their absolute correlation with the target as values
    feature_importance = correlations_with_target.to_dict()
    feature_importance.pop(target_feature, None)

    return feature_importance


def setup_high_cross_correlation_analysis(
        data_cleaned, original_numerical_features, table_placeholder):
    feature_importance = get_feature_importance_from_corr_with_target(data_cleaned[original_numerical_features + [target_feature]])
    cross_corr_cutoff = st.sidebar.slider(
        'Cross-correlation cutoff', min_value=0.0, max_value=1.0, value=0.6,
        step=0.01, on_change=partial(
            update_features_to_exclude_high_cross_corr_from_slider,
            feature_importance=feature_importance,
            data=data_cleaned[original_numerical_features],
            placeholder=table_placeholder), key='cross_corr_cutoff')
    update_features_to_exclude_high_cross_corr_from_slider(feature_importance, data_cleaned[
        original_numerical_features + [target_feature]], table_placeholder)


def setup_target_correlation_analysis(data_cleaned, original_numerical_features, table_placeholder,
                                      multiselect_placeholder):
    # Calculate correlations with the target feature
    target_corr = data_cleaned[original_numerical_features + [target_feature]].corr().abs()[target_feature].drop(
        target_feature)

    # Safeguard for when there's no variability or insufficient data
    if target_corr.empty or target_corr.nunique() == 1:
        st.sidebar.warning('Insufficient data variability for correlation analysis.')
        return

    min_log_value = float(np.log10(target_corr.min())) if target_corr.min() > 0 else -2
    max_log_value = float(np.log10(target_corr.max())) if target_corr.max() > 0 else min_log_value + 1

    # Define a slider for adjusting the target correlation cutoff, using logarithmic scale
    target_corr_cutoff_log10 = st.sidebar.slider(
        'Target correlation cutoff (log scale)',
        min_value=min_log_value,
        max_value=max_log_value,
        value=min_log_value,
        step=0.01,
        key='target_corr_cutoff_log10'
    )
    # Convert from log scale to linear scale for the cutoff
    target_corr_cutoff = 10 ** target_corr_cutoff_log10
    st.session_state['target_corr_cutoff'] = target_corr_cutoff

    # Update the session state for low target correlation features based on the slider
    update_low_target_corr_features_from_slider(target_corr)

    # Multiselect for manually selecting features to exclude due to low correlation with the target
    low_target_corr_features_selection = multiselect_placeholder.multiselect(
        'Select features to exclude due to low correlation with the target',
        options=original_numerical_features,
        default=list(st.session_state.get('low_target_corr_features', [])),
        key='low_target_corr_features_selection'
    )

    # Update session state based on selection
    st.session_state['low_target_corr_features'] = set(low_target_corr_features_selection)

    display_target_correlation_table(data_cleaned, st.session_state['low_target_corr_features'], target_feature,
                                     table_placeholder)


def handle_save_features_and_data_for_modeling(data_cleaned, original_numerical_features):
    if st.sidebar.button('Save Features & Data for Modeling'):
        features_to_exclude = set(st.session_state['features_to_exclude_high_cross_corr']).union(
            st.session_state['low_target_corr_features'])
        data_for_modeling = data_cleaned[[feature for feature in original_numerical_features + [target_feature] if
                                          feature not in features_to_exclude]]
        data_for_modeling.to_csv('data_for_modeling.csv', index=False)
        st.session_state['features_ready_for_modeling'] = True
        st.sidebar.success('Data saved successfully! You can move on to the Model Training step.')
        st.balloons()


def display_scatter_plots_for_discarded_features(
        data_cleaned, target_feature, button_placeholder, charts_placeholder):
    # Button to trigger scatter plots display
    display_low_target_correlations = button_placeholder.button(
        "Display Scatter Plots for Features Discarded Due to Low Correlation with the Target"
    )

    initial_messages = (
        "To display scatter plots of features excluded due to low correlation with the target, "
        "select the features using the slider in the sidebar or the dropdown box above, "
        "and press the button. "
        "NOTE: IT MAY TAKE A WHILE FOR PLOTS TO APPEAR, "
        "ESPECIALLY IF MANY FEATURES ARE SELECTED FOR EXCLUSION.")
    if display_low_target_correlations:
        display_feature_target_scatter_plots(data_cleaned, target_feature, charts_placeholder, initial_messages)
    else:
        charts_placeholder.warning(initial_messages)


def select_features(data_cleaned):
    """
    Identifies and separates numerical and categorical features from the cleaned dataset.

    Parameters:
    - data_cleaned: A pandas DataFrame containing the cleaned data.

    Returns:
    - A tuple containing two lists: one for numerical features and another for categorical features.
    """
    numerical_features = [col for col in data_cleaned.columns if (
            data_cleaned[col].dtype == 'float64' or data_cleaned[col].dtype == 'int64')]
    categorical_features = [col for col in data_cleaned.columns if data_cleaned[col].dtype == 'object']
    return numerical_features, categorical_features


def display_cross_correlations_heatmap(data, placeholder, highlight_features=None):
    """
    Displays a heatmap of cross-correlations among features in the provided dataset.

    Parameters:
    - data: A pandas DataFrame with the data to analyze.
    - placeholder: A Streamlit placeholder for displaying the heatmap.
    - highlight_features: Optional; a set of features to highlight in the heatmap.

    The heatmap helps in identifying highly correlated features that might need to be addressed during feature selection.
    """
    corr = data.corr().reset_index().melt(id_vars='index')
    corr.columns = ['x', 'y', 'value']
    highlight = set(highlight_features) if highlight_features else set()
    corr['x'] = corr['x'].apply(lambda x: x.upper() if x in highlight else x)
    corr['y'] = corr['y'].apply(lambda y: y.upper() if y in highlight else y)
    heatmap = alt.Chart(corr).encode(
        x=alt.X('x:O', axis=alt.Axis(labels=True)),
        y=alt.Y('y:O', axis=alt.Axis(labels=True))
    ).mark_rect().encode(
        color='value:Q',
        tooltip=['x', 'y', 'value']
    ).properties(
        width=800,
        title='Cross-Correlation Heatmap of all Features (cross-correlated features beyond cutoff are CAPITALIZED)'
    ).configure_axis(labelFontSize=9)

    placeholder.altair_chart(heatmap, use_container_width=True)


def display_target_correlation_table(data, discarded_features, target_feature, placeholder):
    """
    Displays a table of features showing their correlation with the target feature.

    Parameters:
    - data: A pandas DataFrame with the dataset including the target feature.
    - discarded_features: A set of features considered for exclusion based on low correlation with the target.
    - target_feature: The name of the target feature for correlation calculation.
    - placeholder: A Streamlit placeholder for displaying the table.

    This table aids in visualizing which features have low correlations with the target and might be excluded from modeling.
    """
    numeric_columns = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col != target_feature]
    corr = data[numeric_columns + [target_feature]].corr()[[target_feature]].drop(target_feature)
    corr.rename(columns={target_feature: 'Correlation with Target'}, inplace=True)
    corr['abs_corr'] = corr['Correlation with Target'].abs()
    corr['log10_abs_corr'] = np.log10(corr['abs_corr'])
    corr['discarded'] = corr.index.isin(discarded_features)
    corr.sort_values(['discarded', 'abs_corr'], ascending=[False, True], inplace=True)

    def color_discarded(val):
        color = 'gray' if val else 'black'
        return f'color: {color}'

    formatted_corr = corr.style.applymap(color_discarded, subset=['discarded']) \
        .format({'discarded': lambda x: 'Yes' if x else 'No',
                 'Correlation with Target': "{:.4f}",
                 'abs_corr': "{:.4f}",
                 'log10_abs_corr': "{:.4f}"})

    if not corr.empty:
        placeholder.write(
            "Correlations of selected features to discard due to low correlation to the target variable:")
        display_dataframe_in_placeholder(placeholder, formatted_corr)
    else:
        placeholder.warning("Use the multiselect widget above to select features for exclusion.")


def display_feature_target_scatter_plots(data_cleaned, target_feature, placeholder, initial_message):
    if 'low_target_corr_features' in st.session_state and st.session_state['low_target_corr_features']:
        discarded_features = st.session_state['low_target_corr_features']
        features_to_display = list(discarded_features) + [target_feature]
        discarded_data = data_cleaned[features_to_display]

        # List of features excluding the target feature
        features = discarded_data.columns.drop(target_feature)

        # Create a figure with subplots
        fig, axs = plt.subplots(len(features), 1, figsize=(8, 5 * len(features)))

        # If there's only one feature, axs may not be an array
        if len(features) == 1:
            axs = [axs]

        for i, feature in enumerate(features):
            sns.scatterplot(data=discarded_data, x=feature, y=target_feature, ax=axs[i])
            axs[i].set_title(f'{feature} vs. {target_feature}')
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel(target_feature)

        placeholder.markdown(
            "### Wait for plots to appear ... (Scatter Plots for Features Discarded Due to Low Correlation with the Target)")
        placeholder.pyplot(fig)
    else:
        placeholder.warning(initial_message)


def update_features_to_exclude_high_cross_corr_from_slider(feature_importance, data, placeholder):
    features_to_discard = set()
    remaining_features = list(data.columns)

    # Initialize a flag to track if at least one feature has been discarded in the loop
    feature_discarded = True

    while feature_discarded:
        corr_matrix = data[remaining_features].corr().abs()
        high_cross_corr_pairs_series = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
        high_cross_corr_pairs = high_cross_corr_pairs_series.to_frame('corr').reset_index()
        high_cross_corr_pairs.columns = ['feature_1', 'feature_2', 'corr']
        high_cross_corr_pairs.sort_values('corr', ascending=False, inplace=True)

        # Reset flag to false at the start of each iteration
        feature_discarded = False

        for index, row in high_cross_corr_pairs.iterrows():
            if row['corr'] > st.session_state.cross_corr_cutoff:
                # If the feature importance is NaN (defaulting to -1), consider it unimportant and discard
                importance_feature_1 = feature_importance.get(row['feature_1'], -1)
                importance_feature_2 = feature_importance.get(row['feature_2'], -1)

                if importance_feature_1 == -1 or importance_feature_2 == -1:
                    # If either of the feature importances is NaN, discard the one with NaN importance
                    feature_to_discard = row['feature_1'] if importance_feature_1 == -1 else row['feature_2']
                else:
                    # If neither importance is NaN, discard the one with lower importance
                    feature_to_discard = row['feature_2'] if importance_feature_1 >= importance_feature_2 else row['feature_1']

                features_to_discard.add(feature_to_discard)
                remaining_features.remove(feature_to_discard)

                # Set flag to true as a feature has been discarded
                feature_discarded = True
                break

        if not feature_discarded:
            # Break the loop if no features were discarded in this iteration
            break

    st.session_state['features_to_exclude_high_cross_corr'] = features_to_discard
    display_high_cross_corr_pairs(features_to_discard, placeholder)


def update_low_target_corr_features_from_selection():
    st.session_state['low_target_corr_features'] = set(st.session_state.low_target_corr_features_selection)


def update_low_target_corr_features_from_slider(target_corr):
    st.session_state['low_target_corr_features'] = set(
        target_corr[target_corr < 10 ** st.session_state.target_corr_cutoff_log10].index.tolist())


def display_high_cross_corr_pairs(features_to_discard_set, placeholder):
    data_cleaned = st.session_state['data_cleaned']
    original_numerical_features = [col for col in data_cleaned.columns if
                                   data_cleaned[col].dtype in ['float64', 'int64']]

    if features_to_discard_set:
        # Labels of features to discard due to high cross-correlation
        # are highlighted in the chart above and listed below in the multiselect tool,
        # which you can use to alter the list manually
        display_cross_correlations_heatmap(
            data_cleaned[original_numerical_features], placeholder,
            highlight_features=features_to_discard_set)
    else:
        placeholder.warning(
            'Select the highly cross-correlated features to exclude them from modeling'
            'by using the slider in the sidebar and adjusting your selection manually '
            'in the corresponding multiselect tool.')
