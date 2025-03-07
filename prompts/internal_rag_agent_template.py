def generate_instructions_template():
    """
    You are an advanced tool designed for structured data extraction and contextual analysis. Your task is to process the given text and accurately identify key elements related to dataset types and explanation methods.

    Your response should capture:
    - **Dataset Types**: Recognizing different types of datasets mentioned in the text.
    - **Explainers**: Extracting references to explanation techniques such as 'LIME', 'SHAP', 'Anchors', etc.

    Given the following text:

    {$input_text}

    Identify the types of datasets being discussed (e.g., 'Multivariate tabular', 'Multivariate time series', 'Univariate time series', 'Image', 'Text', etc.) and list the names of explainers mentioned.

    Provide the output in the following JSON format:

    {{
        "DatasetTypes": ["dataset_type_1", "dataset_type_2", ...],
        "Explainers": ["explainer_name_1", "explainer_name_2", ...]
    }}

    {$few_shot_examples}

    Your response:
    """
    return generate_instructions_template.__doc__
