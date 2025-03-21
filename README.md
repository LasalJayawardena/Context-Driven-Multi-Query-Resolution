# Context Driven Multi-Query Resolution using LLM-RAG to support the Revision of Explainability needs

This repository contains code and assets related to the paper titled "Context Driven Multi-Query Resolution using LLM-RAG to support the Revision of Explainability needs". 

## Keywords

LLMs, CBR, RAG, Q&A, LLM-as-a-Judge, Behaviour Trees, Case Attributes, Intent Evolution

## Repository Structure
```
├── Analytics_Notebook.ipynb
├── Evaluation_Notebook.ipynb
├── README.md
├── assets
│   ├── cluster_lexical
│   │   ├── cluster_scatter_10.html
│   │   ├── cluster_scatter_2.html
│   │   ├── cluster_scatter_3.html
│   │   ├── cluster_scatter_4.html
│   │   ├── cluster_scatter_5.html
│   │   ├── cluster_scatter_6.html
│   │   ├── cluster_scatter_7.html
│   │   ├── cluster_scatter_8.html
│   │   ├── cluster_scatter_9.html
│   │   ├── elbow_plot.html
│   │   ├── final_cluster_data.csv
│   │   ├── ideal_cluster_scatter.html
│   │   ├── silhouette_plot.html
│   │   ├── user_segmentation_k_10.html
│   │   ├── user_segmentation_k_2.html
│   │   ├── user_segmentation_k_3.html
│   │   ├── user_segmentation_k_4.html
│   │   ├── user_segmentation_k_5.html
│   │   ├── user_segmentation_k_6.html
│   │   ├── user_segmentation_k_7.html
│   │   ├── user_segmentation_k_8.html
│   │   └── user_segmentation_k_9.html
│   ├── cluster_lexical_readability
│   │   ├── cluster_scatter_10.html
│   │   ├── cluster_scatter_2.html
│   │   ├── cluster_scatter_3.html
│   │   ├── cluster_scatter_4.html
│   │   ├── cluster_scatter_5.html
│   │   ├── cluster_scatter_6.html
│   │   ├── cluster_scatter_7.html
│   │   ├── cluster_scatter_8.html
│   │   ├── cluster_scatter_9.html
│   │   ├── elbow_plot.html
│   │   ├── final_cluster_data.csv
│   │   ├── ideal_cluster_scatter.html
│   │   ├── silhouette_plot.html
│   │   ├── user_segmentation_k_10.html
│   │   ├── user_segmentation_k_2.html
│   │   ├── user_segmentation_k_3.html
│   │   ├── user_segmentation_k_4.html
│   │   ├── user_segmentation_k_5.html
│   │   ├── user_segmentation_k_6.html
│   │   ├── user_segmentation_k_7.html
│   │   ├── user_segmentation_k_8.html
│   │   └── user_segmentation_k_9.html
│   ├── cluster_semantic
│   │   ├── cluster_scatter_10.html
│   │   ├── cluster_scatter_2.html
│   │   ├── cluster_scatter_3.html
│   │   ├── cluster_scatter_4.html
│   │   ├── cluster_scatter_5.html
│   │   ├── cluster_scatter_6.html
│   │   ├── cluster_scatter_7.html
│   │   ├── cluster_scatter_8.html
│   │   ├── cluster_scatter_9.html
│   │   ├── elbow_plot.html
│   │   ├── final_cluster_data.csv
│   │   ├── ideal_cluster_scatter.html
│   │   ├── silhouette_plot.html
│   │   ├── user_segmentation_k_10.html
│   │   ├── user_segmentation_k_2.html
│   │   ├── user_segmentation_k_3.html
│   │   ├── user_segmentation_k_4.html
│   │   ├── user_segmentation_k_5.html
│   │   ├── user_segmentation_k_6.html
│   │   ├── user_segmentation_k_7.html
│   │   ├── user_segmentation_k_8.html
│   │   └── user_segmentation_k_9.html
│   ├── full_correlation.png
│   ├── llm_reasonability_advanced_stats
│   │   ├── impartial_output.json
│   │   ├── question_stats_table.html
│   │   └── stats_output.json
│   ├── llm_reasonability_scores
│   │   ├── anthropic_claude_2.0_reasonability_dict.json
│   │   ├── anthropic_claude_2.1_reasonability_dict.json
│   │   ├── anthropic_claude_3.5_haiku_reasonability_dict.json
│   │   ├── anthropic_claude_3.5_sonnet_reasonability_dict.json
│   │   ├── anthropic_claude_3_haiku_reasonability_dict.json
│   │   ├── anthropic_claude_3_opus_reasonability_dict.json
│   │   ├── anthropic_claude_3_sonnet_reasonability_dict.json
│   │   ├── deepseek_llama_8b_reasonability_dict.json
│   │   ├── deepseek_qwen_1.5b_reasonability_dict.json
│   │   ├── deepseek_qwen_14b_reasonability_dict.json
│   │   ├── deepseek_qwen_32b_reasonability_dict.json
│   │   ├── deepseek_qwen_7b_reasonability_dict.json
│   │   ├── deepseek_v2_16b_reasonability_dict.json
│   │   ├── falcon3_10b_reasonability_dict.json
│   │   ├── falcon3_3b_reasonability_dict.json
│   │   ├── falcon3_7b_reasonability_dict.json
│   │   ├── gemini_2_flash_lite_reasonability_dict.json
│   │   ├── gemini_2_flash_reasonability_dict.json
│   │   ├── gemini_2_flash_thinking_reasonability_dict.json
│   │   ├── gemini_2_pro_reasonability_dict.json
│   │   ├── gemini_flash_1.5_8b_reasonability_dict.json
│   │   ├── gemini_flash_1.5_reasonability_dict.json
│   │   ├── gemini_learnlm_1.5_reasonability_dict.json
│   │   ├── gemini_pro_1.0_reasonability_dict.json
│   │   ├── gemini_pro_1.5_reasonability_dict.json
│   │   ├── gemma2_27b_reasonability_dict.json
│   │   ├── gemma2_2b_reasonability_dict.json
│   │   ├── gemma2_9b_reasonability_dict.json
│   │   ├── gemma_7b_reasonability_dict.json
│   │   ├── granite_3.1_2b_reasonability_dict.json
│   │   ├── granite_3.1_8b_reasonability_dict.json
│   │   ├── granite_3.2_8b_reasonability_dict.json
│   │   ├── granite_3_2b_reasonability_dict.json
│   │   ├── granite_3_8b_reasonability_dict.json
│   │   ├── llama_2_13b_reasonability_dict.json
│   │   ├── llama_31_8b_reasonability_dict.json
│   │   ├── llama_32_1b_reasonability_dict.json
│   │   ├── llama_32_3b_reasonability_dict.json
│   │   ├── llama_3_8b_reasonability_dict.json
│   │   ├── minimistral_3b_reasonability_dict.json
│   │   ├── minimistral_8b_reasonability_dict.json
│   │   ├── mistral_7b_reasonability_dict.json
│   │   ├── mistral_large_reasonability_dict.json
│   │   ├── mistral_medium_reasonability_dict.json
│   │   ├── mistral_nemo_12b_reasonability_dict.json
│   │   ├── mistral_nemo_reasonability_dict.json
│   │   ├── mistral_saba_reasonability_dict.json
│   │   ├── mistral_small_reasonability_dict.json
│   │   ├── mixtral_8x22b_reasonability_dict.json
│   │   ├── mixtral_8x7b_reasonability_dict.json
│   │   ├── nemotron_mini_4b_reasonability_dict.json
│   │   ├── nvidia_chatqa_8b_reasonability_dict.json
│   │   ├── openai_gpt_3.5_turbo_reasonability_dict.json
│   │   ├── openai_gpt_4_reasonability_dict.json
│   │   ├── openai_gpt_4_turbo_reasonability_dict.json
│   │   ├── openai_gpt_4o_mini_reasonability_dict.json
│   │   ├── openai_gpt_4o_reasonability_dict.json
│   │   ├── openai_gpt_o1_mini_reasonability_dict.json
│   │   ├── openai_gpt_o1_reasonability_dict.json
│   │   ├── openai_gpt_o3_mini_reasonability_dict.json
│   │   ├── phi_4_14b_reasonability_dict.json
│   │   ├── qwen_2.5_14b_reasonability_dict.json
│   │   ├── qwen_2.5_32b_reasonability_dict.json
│   │   ├── qwen_2.5_3b_reasonability_dict.json
│   │   ├── qwen_2.5_7b_reasonability_dict.json
│   │   ├── qwen_2_1.5b_reasonability_dict.json
│   │   ├── qwen_32b_reasonability_dict.json
│   │   ├── qwen_7b_reasonability_dict.json
│   │   ├── vicuna_13b_reasonability_dict.json
│   │   └── vicuna_7b_reasonability_dict.json
│   ├── llm_reasonability_stats
│   │   ├── explanations_output.json
│   │   ├── question_data_output.json
│   │   ├── scores_output.json
│   │   ├── stats_average.json
│   │   └── stats_output.json
│   └── sentiment_analysis
│       ├── answer_llm_by_explainer.html
│       ├── answer_llm_by_user_persona.html
│       ├── answer_vader_by_usecase.html
│       ├── compare_answer_sentiments.html
│       ├── compare_question_sentiments.html
│       ├── question_llm_by_explainer.html
│       ├── question_llm_by_usecase.html
│       ├── question_vader_by_user_persona.html
│       ├── reasonability_vs_answer_llm.html
│       ├── reasonability_vs_question_llm.html
│       └── sentiment_dataset.csv
├── data
│   ├── chat_logs
│   │   |── chat_history_2024-12-10_14-55-00-checkpoint.json
│   │   ├── chat_history_2024-12-10_14-55-00.json
│   │   ├── chat_history_2024-12-11_08-07-05.json
│   │   ├── chat_history_2024-12-12_08-16-47.json
│   │   ├── chat_history_2024-12-13_10-11-29.json
│   │   ├── chat_history_2024-12-15_12-31-56.json
│   │   ├── chat_history_2025-01-06_02-20-55.json
│   │   ├── chat_history_2025-01-06_02-23-29.json
│   │   ├── chat_history_2025-01-06_02-26-15.json
│   │   ├── chat_history_2025-01-06_02-30-13.json
│   │   ├── chat_history_2025-01-06_02-33-49.json
│   │   ├── chat_history_2025-01-06_02-35-07.json
│   │   ├── chat_history_2025-01-06_02-39-57.json
│   │   ├── chat_history_2025-01-06_13-55-29.json
│   │   ├── chat_history_2025-01-06_13-59-22.json
│   │   ├── chat_history_2025-01-06_14-00-25.json
│   │   ├── chat_history_2025-01-06_14-06-17.json
│   │   ├── chat_history_2025-01-06_14-14-06.json
│   │   ├── chat_history_2025-01-06_14-30-56.json
│   │   ├── chat_history_2025-01-06_15-05-27.json
│   │   ├── chat_history_2025-01-06_16-34-05.json
│   │   ├── chat_history_2025-01-09_16-52-50.json
│   │   ├── chat_history_2025-01-09_17-03-45.json
│   │   ├── chat_history_2025-01-09_17-12-50.json
│   │   ├── chat_history_2025-01-09_17-24-45.json
│   │   ├── chat_history_2025-01-09_17-27-41.json
│   │   ├── chat_history_2025-01-10_10-49-09.json
│   │   ├── chat_history_2025-01-10_16-14-18.json
│   │   ├── chat_history_2025-01-14_14-03-28.json
│   │   └── chat_history_2025-01-14_14-07-38.json
│   ├── html_context.json
│   ├── intents_used_results_with_exp.json
│   ├── isee_explainer_categories.json
│   ├── isee_explainer_names.json
│   ├── llm_model_info.json
│   ├── question_emb_dict.json
│   ├── user_interaction_list.json
│   └── user_interaction_rich_data.json
├── prompts
│   ├── bt_qa_agent_template].py
│   ├── internal_rag_agent_template.py
│   ├── llm_sentiment_analysis_template.py
│   ├── reasonability_score_template.py
│   └── response_enhancer_agent_template.py
└── utils
    ├── bt_utils.py
    ├── llm_base.py
    └── user_interaction_assets.py
```


## Citation

If you find this work helpful or use any part of the provided code or datasets, please consider citing the original paper: TBA
