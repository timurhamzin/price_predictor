import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_dataframe_in_placeholder(placeholder, dataframe):
    """
    Attempts to display a dataframe in a specified Streamlit placeholder.
    Falls back to displaying the dataframe as strings in case of an exception.

    Parameters:
    - placeholder: Streamlit container or placeholder object where the dataframe will be displayed.
    - dataframe: DataFrame, the pandas DataFrame to be displayed.
    """
    try:
        placeholder.dataframe(dataframe)
    except Exception as e:
        logger.error('Failed to display dataframe using Streamlit dataframe method. Error: %s', str(e))
        logger.info('Displaying dataframe as strings as a fallback.')
        placeholder.dataframe(dataframe.astype(str))
