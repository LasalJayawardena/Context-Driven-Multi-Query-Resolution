# Import Libraries
import os
import json
import re
import requests
from tqdm import tqdm

# Main Code

## Extract Funtions
def extract_usecase_name(text):
    pattern = r"iSee Chatbot for (.*?), Would you like to proceed\?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_llm_segment(bt_tree):
    llm_process_list = []
    execution_branch = bt_tree["executions"]
    for execution in execution_branch:
        if "CLR_EXEC" in execution:
            # print(execution)
            process = execution["CLR_EXEC"]["<CLR_EXEC>"]
            llm_process_list.append(process)

    return llm_process_list

def extract_explainer_name_segment(bt_tree):
    explainer_name_list = []
    node_branch = bt_tree["nodes"]
    for node in node_branch:
        BT_Tree_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#properties"
        BT_Tree_node_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#hasDictionaryMember"
        BT_Tree_explainer_action_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#pairKey"
        BT_Tree_explainer_name_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#pair_value_object"
        
        if BT_Tree_id in node:
            # print(node)
            if BT_Tree_node_id in node[BT_Tree_id]:
                possible_nodes = node[BT_Tree_id][BT_Tree_node_id]
                for possible_node in possible_nodes:
                    # print(possible_node)
                    if (BT_Tree_explainer_action_id in possible_node):
                        if possible_node[BT_Tree_explainer_action_id] == "endpoint":
                            explainer_name = possible_node[BT_Tree_explainer_name_id]
                            explainer_name_list.append(explainer_name)

    return explainer_name_list

def extract_base64_data(data_url):
    base64_pattern = r'^data:image/[a-zA-Z]+;base64,(.+)$'
    match = re.match(base64_pattern, data_url)
    if match:
        return match.group(1)
    else:
        return None

def extract_llm_context(bt_tree):
    interaction_list = []
    execution_branch = bt_tree["executions"]
    for execution in execution_branch:
        BT_Tree_id = "http://www.w3.org/ns/prov#generated"
        BT_Tree_Property_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#properties"
        BT_Tree_Property_Dict_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#hasDictionaryMember"
        BT_Tree_Question_Type_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#pairKey"
        BT_Tree_Value_id = "https://www.w3id.org/iSeeOnto/BehaviourTree#pair_value_object"
        
        
        if BT_Tree_id in execution:
            if BT_Tree_Property_id in execution[BT_Tree_id]:
                execution_property_info = execution[BT_Tree_id][BT_Tree_Property_id][BT_Tree_Property_Dict_id]
                if len(execution_property_info) > 1:
                    if len(execution_property_info) == 2:
                        query, response = execution_property_info 

                        query_type = query[BT_Tree_Question_Type_id]
                        query = query[BT_Tree_Value_id]
                        response = response[BT_Tree_Value_id]

                        if "content" in  query:
                            query_content = query["content"]
                            
                            if "data:image" in query_content: 
                                start_index = query_content.find('src="data:image')
                                end_index = query_content.find('"', start_index + 5)
                                query_content = query_content[:start_index] + "<Base64-Encoded-Image-Content" + query_content[end_index:]
                            
                            # print(query_content)
                            if query.get("responseType") == "Radio":
                                query_options = [x["content"] for x in query["responseOptions"]]
                            else: 
                                query_options = None
                        else:
                            query_content = None

                        if (response.get("id") == "okay") or (response.get("id") == "temp"):
                            response_content = None
                        else:
                            response_content = response.get("content")
                        
                        interaction_list.append([response_content, query_content, query_options])

    # print(len(interaction_list))
    interaction_context = "LLM Context for Claririfictaion Questions:\n========= \n"

    for (response_content, query_content, query_options) in interaction_list:
        # interaction_context += "---------\n"
        interaction_context += f"<AI Dialog Manager: > {query_content}\n"
        interaction_context += f"<Question Options for Query: > {query_options}\n" if query_options is not None else ""
        # interaction_context += f"<User Selection: > {query_content}\n"
        interaction_context += f"<User Response: > {response_content}\n" if response_content is not None else "<User Response: > <<Interacted>>\n"
        interaction_context += "---------\n"
                    

    return interaction_list, interaction_context


# Run Code
if __name__ == "__main__":

    chat_dir = "../data/chat_logs/"

    chat_json_dict, chat_text_dict = {}, {}
    chat_id_list = []

    for file in os.listdir(chat_dir):
        if "json" not in file:
            continue
        chat_id = file.split(".")[0]
        chat_id_list.append(chat_id)
        
        with open(f"{chat_dir}/{file}") as f:
            chat_bt_content = json.load(f)
            # chat_text_content = str(chat_bt_content)
            chat_text_content = json.dumps(chat_bt_content)

        chat_json_dict[chat_id] = chat_bt_content
        chat_text_dict[chat_id] = chat_text_content

    ## Isee Explainer list
    isee_explainer_data = json.loads(requests.get("https://api-onto-dev.isee4xai.com/api/explainers/list").content)
    isee_explainer_names = [x['name'] for x in isee_explainer_data]
    isee_explainer_categories = list(set([x.split("/")[1] for x in isee_explainer_names]))

    user_interaction_list = list(chat_json_dict.keys())
    user_interaction_rich_data = {}

    for interaction in tqdm(user_interaction_list):
        usecase_name = extract_usecase_name(chat_text_dict[interaction])
        clarifictaion_segment = extract_llm_segment(chat_json_dict[interaction])
        explainers_used = extract_explainer_name_segment(chat_json_dict[interaction])
        bt_interaction_list, llm_interaction_context = extract_llm_context(chat_json_dict[interaction])

        if usecase_name == "Fracture Detection (Jiva)":
            usecase_name = "Jiva Fracture Detection System"

        user_interaction_rich_data[interaction] = {
            "usecase_name": usecase_name,
            "explainers_used":explainers_used,
            "bt_interaction_list": bt_interaction_list,
            "llm_interaction_context": llm_interaction_context,
            "clarifictaion_segment": clarifictaion_segment,
        }

    # Save to Data Folder
    with open("../data/user_interaction_list.json", "w") as f:
        json.dump(user_interaction_list, f)

    with open("../data/user_interaction_rich_data.json", "w") as f:
        json.dump(user_interaction_rich_data, f)
   
    with open("../data/isee_explainer_names.json", "w") as f:
        json.dump(isee_explainer_names, f)

    with open("../data/isee_explainer_categories.json", "w") as f:
        json.dump(isee_explainer_categories, f)
