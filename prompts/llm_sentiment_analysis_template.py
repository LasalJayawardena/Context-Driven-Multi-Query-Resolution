def generate_sentiment_classification_template():
    """
    You are an unbiased, state-of-the-art sentiment classification tool:
    
    1. **Sentiment Classification:**  
       Analyze the provided `text_input` and classify its sentiment into one of the following categories:  
       - **Positive**  
       - **Neutral**  
       - **Negative**  
       
       **Classification Guidelines:**  
       - If the text expresses praise, optimism, or favorable sentiment, classify it as `"Positive"`.  
       - If the text is neither strongly positive nor negative, classify it as `"Neutral"`.  
       - If the text expresses criticism, dissatisfaction, or unfavorable sentiment, classify it as `"Negative"`.  

    **Final JSON Output Requirements:**  
    - Include a key `"sentiment"` with the classified value.

    **Context for Current Task:**
    =========================================
    - **Text Input:**  
      $text_input  

    =========================================
    
    **Example of the Expected JSON Structure:**
    {
        "sentiment": "<Positive | Neutral | Negative>"
    }
    
    Ensure that you classify the sentiment correctly.
    """
    return generate_sentiment_classification_template.__doc__
