from string import Template
import json
import re
import markdown
import json
from collections import OrderedDict, defaultdict
from string import Template

import os
import time

from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables (or read .env file)
load_dotenv()

def extract_node_properties_and_images(json_data):
    # Parse the JSON data
    data = json.loads(json_data) if type(json_data) != dict else json_data
    
    # Extract the executions
    executions = data.get('executions', [])
    
    # Dictionary to store image references
    image_references = {}
    image_counter = 1
    
    # List to store node properties as text in order of execution
    node_properties_text = []
    
    # List to store explanations given
    explanations_given = []
    
    for execution in executions:
        node_instance = execution.get('https://www.w3id.org/iSeeOnto/BehaviourTreeExecution#enacted', {}).get('instance')
        node_properties = None
        
        for node in data.get('nodes', []):
            if node.get('instance') == node_instance:
                node_properties = node.get('https://www.w3id.org/iSeeOnto/BehaviourTree#properties', {}).get('https://www.w3id.org/iSeeOnto/BehaviourTree#hasDictionaryMember', [])
                break
        
        if node_properties:
            properties_text = f"Node Instance: {node_instance}\n"
            for prop in node_properties:
                key = prop.get('https://www.w3id.org/iSeeOnto/BehaviourTree#pairKey')
                value = prop.get('https://www.w3id.org/iSeeOnto/BehaviourTree#pair_value_object')
                
                properties_text += f"  {key}: {value}\n"
                
                # Check if the key is 'question' to extract explanations
                if key == 'question':
                    explanations_given.append(value)
            
            # Add execution content
            node_exec = execution.get("http://www.w3.org/ns/prov#generated", {})
            if "https://www.w3id.org/iSeeOnto/BehaviourTree#properties" in node_exec:
                exec_properties = node_exec["https://www.w3id.org/iSeeOnto/BehaviourTree#properties"]
                exec_properties_text = "  Execution Properties:\n"
                for exec_prop in exec_properties.get("https://www.w3id.org/iSeeOnto/BehaviourTree#hasDictionaryMember", []):
                    exec_key = exec_prop.get('https://www.w3id.org/iSeeOnto/BehaviourTree#pairKey')
                    exec_value = exec_prop.get('https://www.w3id.org/iSeeOnto/BehaviourTree#pair_value_object')
                    
                    # Check if the value is an image
                    if isinstance(exec_value.get("content"), str) and 'src="data:image' in exec_value:
                        # Extract the base64 image data from the src attribute
                        start_index = exec_value.find('src="data:image')
                        end_index = exec_value.find('"', start_index + 5)
                        image_data = exec_value[start_index + 5:end_index]
                        image_placeholder = f"image[{image_counter}]"
                        image_references[image_counter] = image_data
                        exec_properties_text += f"    {exec_key}: {image_placeholder}\n"
                        image_counter += 1
                    else:
                        exec_properties_text += f"    {exec_key}: {exec_value}\n"
                
                properties_text += exec_properties_text
            
            node_properties_text.append(properties_text)
    
    return "\n".join(node_properties_text), image_references, explanations_given

# Function to extract base64 data from image URLs
def extract_base64_data(data_url):
    base64_pattern = r'^data:image/[a-zA-Z]+;base64,(.+)$'
    match = re.match(base64_pattern, data_url)
    if match:
        return match.group(1)
    else:
        return None


def clean_and_convert_to_html(content: str) -> str:
    """
    Cleans the input content by removing code fences or markers (e.g., ```html, ```markdown, ```whatever)
    from the beginning and end of the string. Detects if the content is Markdown or HTML, and converts
    Markdown content to HTML.

    Args:
        content (str): The input string containing code fences and either Markdown or HTML content.

    Returns:
        str: The cleaned content as HTML.

    Examples:
        >>> markdown_content = '''```markdown
        ... # Header
        ... This is a *Markdown* example.
        ... ```'''
        >>> clean_and_convert_to_html(markdown_content)
        '<h1>Header</h1>\\n<p>This is a <em>Markdown</em> example.</p>'

        >>> html_content = '''```html
        ... <p>This is valid HTML content.</p>
        ... ```'''
        >>> clean_and_convert_to_html(html_content)
        '<p>This is valid HTML content.</p>'
    """
    # Strip leading and trailing whitespace
    content = content.strip()

    # Remove starting code fence and any language specifier
    content = re.sub(r'^```[\w\s]*\n', '', content)

    # Remove ending code fence
    content = re.sub(r'\n```$', '', content)

    # Strip again in case there was whitespace after removing code fences
    cleaned_content = content.strip()

    # Check if the content contains HTML tags
    def contains_html_tags(text: str) -> bool:
        """
        Checks if the text contains HTML tags.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if HTML tags are found, False otherwise.
        """
        html_tag_pattern = re.compile(r'<[^>]+>')
        return bool(html_tag_pattern.search(text))

    if contains_html_tags(cleaned_content):
        # Assume it's already valid HTML
        return cleaned_content
    else:
        # Convert Markdown to HTML
        return markdown.markdown(cleaned_content)


# Function to extract node properties, images, and explanations
def extract_rich_properties(json_data):
    # Get node properties text and explanations
    node_properties_text, _, explanations_given = extract_node_properties_and_images(json_data)
    # Get image references and their base64 data
    image_references = extract_images_only(json_data)
    image_data_references = {k: extract_base64_data(v) for k,v in image_references.items()}
    explanations_given = [x for x in explanations_given if x != "None"]

    image_ref_names = {}
    for idx, ref in enumerate(list(image_references.keys())):
        image_ref_names[idx + 1] = ([f"<Image-{idx + 1}-URL>", f"<Image-{idx + 1}-Base_Data>"])
        image_url = image_references[ref]
        image_base_data = image_data_references[ref]
        node_properties_text = node_properties_text.replace(image_url, f"<Image-{idx + 1}-URL>").replace(image_base_data, f"<Image-{idx + 1}-Base_Data>")

    return node_properties_text, explanations_given, image_references, image_data_references, image_ref_names

# Convert image references to text
def convert_image_references_to_text(image_references):
    text_output = []
    for idx, (image_number, references) in enumerate(image_references.items(), start=1):
        url, base64_data = references
        text = f"The {ordinal(idx)} image URL is referenced by this: {url}, and its base64 encoded data is referenced like this: {base64_data}."
        text_output.append(text)
    return " ".join(text_output)

# Helper function to convert number to ordinal
def ordinal(n):
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

# Convert chat history to text
def convert_chat_history_to_text(chat_history):
    formatted_history = []
    for entry in chat_history:
        if "user" in entry:
            formatted_history.append(f"User: {entry['user']}")
        elif "system" in entry:
            formatted_history.append(f"System: {entry['system']}")
    return "\n".join(formatted_history) if len(chat_history) != 0 else "[]"

# Function to group and track order of clarification nodes
def process_executions_with_order(data):
    executions = data.get("executions", [])
    node_history = defaultdict(list)  # Dictionary to store history for each node_id
    latest_node_entry = {}  # Store the latest entry for each node_id
    node_order = []  # List to keep track of clarification node order

    # Iterate through each execution in the list
    for index, exec_data in enumerate(executions):
        clr_exec = exec_data.get("CLR_EXEC", {}).get("<CLR_EXEC>", [])
        
        # Find the clarification_node_id in this execution group
        clarification_node_id = next(
            (item['clarification_node_id'] for item in clr_exec if 'clarification_node_id' in item), None)

        if clarification_node_id:
            # Append the entire execution group to the node's history
            node_history[clarification_node_id].append(exec_data)
            
            # Update the latest node group and entry (assuming the most recent one is the last)
            latest_node_entry[clarification_node_id] = {
                "clarification_question": next((item['clarification_question'] for item in clr_exec if 'clarification_question' in item), None),
                "llm_response": next((item['llm_response'] for item in clr_exec if 'llm_response' in item), None),
                "llm_history": next((item['llm_history'] for item in clr_exec if 'llm_history' in item), None),
                "clarification_node_id": clarification_node_id
            }
            
            # Add the clarification node and its position in the order
            node_order.append({
                "clarification_node_id": clarification_node_id,
                "execution_order": index + 1  # Tracking the execution order (1-based)
            })

    # Now get the last node based on order
    if node_order:
        last_node_id = node_order[-1]['clarification_node_id']  # Last node in the order
        last_node_history = node_history[last_node_id]  # Get all execution data for the last node

        # Create the final dictionary with latest node group, latest node, full node history, and order
        result = {
            "last_node_id": last_node_id,
            "last_node_history": last_node_history,  # All executions for the last node
            "latest_node_entry": latest_node_entry,  # Latest entry by node_id
            "full_node_history": node_history,       # Full history by node_id
            "node_order": node_order                 # Order of clarification nodes in executions
        }
    else:
        result = {
            "last_node_id": None,
            "last_node_history": [],
            "latest_node_entry": latest_node_entry,
            "full_node_history": node_history,
            "node_order": node_order
        }

    # Iterate through each execution in the list
    last_node_group = []
    for index, exec_data in enumerate(executions):
        clr_exec = exec_data.get("CLR_EXEC", {}).get("<CLR_EXEC>", [])
        
        # Find the clarification_node_id in this execution group
        clarification_node_id = next(
            (item['clarification_node_id'] for item in clr_exec if 'clarification_node_id' in item), None)

        if clarification_node_id:
            # Group chat history of last explainer
            if clarification_node_id == last_node_id:
                last_node_group.append(exec_data)

    result["latest_node_group"] = last_node_group
    return result

# Template for the overall clarification history
clarification_template = Template("""
Clarification Node:
$clarification_node

Clarification History:
$user_conversation
""")

# Template for each user and assistant conversation
conversation_template = Template("""
<user>: $user_question
<assistant>: $llm_response
""")

# Function to generate clarification conversation history
from string import Template

# Template for the overall clarification history
clarification_template = Template("""
Clarification Node:
$clarification_node

Clarification History:
$user_conversation
""")

# Template for each user and assistant conversation
conversation_template = Template("""
<user>: $user_question
<assistant>: $llm_response
""")

# Function to generate clarification conversation history
def generate_clarification_history(latest_clarification_group):
    clarification_node = "Unknown Node"
    conversation_history = ""

    # Loop through each entry in the clarification group (assuming it's a list)
    for entry in latest_clarification_group:
        # Directly access the clarification details from CLR_EXEC
        if 'CLR_EXEC' in entry:
            clarification = entry['CLR_EXEC']['<CLR_EXEC>']  # Access the list directly

            # print(clarification[0])
            # Extract clarification details
            clarification_question = clarification[0].get('clarification_question', {})
            llm_response = clarification[1].get('llm_response', 'No Response')
            clarification_node = clarification[3].get('clarification_node_id', clarification_node)
            
            # Build the conversation history using the template
            conversation_history += conversation_template.substitute(
                user_question=clarification_question,
                llm_response=llm_response
            )

    # Fill in the clarification history template
    clarification_history = clarification_template.substitute(
        clarification_node=clarification_node,
        user_conversation=conversation_history
    )

    return clarification_history

import json
from typing import Dict
import data.parser as parser
import business.bt.nodes.node as node
import business.bt.nodes.factory as node_factory
import business.bt.bt as bt
from business.bt.nodes.action import GreeterNode, ExplainerNode
from business.bt.clarification_node import LLMClarificationQuestionNode, RepeatUntilNode # Import the RepeatUntilNode

# Define the node types that can have the LLMClarificationQuestionNode inserted after them
# nodes_that_allow_clarification = [GreeterNode, ExplainerNode] 
nodes_that_allow_clarification = [ExplainerNode] 

def generate_tree_from_file(path, co):
    _parser = parser.TreeFileParser(path)
    return generate_tree(_parser, co)

def generate_tree_from_obj(obj, co):
    _parser = parser.TreeObjParser(obj)
    return generate_tree(_parser, co)

def generate_tree(parser, co):
    nodes: Dict[str, node.Node] = {}

    # First, create all nodes
    for node_id in parser.bt_nodes:
        type_name = parser.bt_nodes[node_id]["Concept"]
        id = parser.bt_nodes[node_id]["id"]
        label = parser.bt_nodes[node_id]["Instance"]

        # Create Node according to its type with factory
        currentNode = node_factory.makeNode(type_name, id, label)
        nodes[node_id] = currentNode

    # Link the nodes together and insert clarification nodes where needed
    for n in parser.bt_nodes:
        nodes.get(n).co = co

        if parser.bt_nodes[n]["Concept"] in ["Priority", "Sequence", "Replacement", "Variant", "Complement", "Supplement"]:
            children = parser.bt_nodes[n]["firstChild"]
            previous_node = None
            while children is not None:
                child_node = nodes.get(children["Id"])

                # Attach the original child node
                nodes.get(n).children.append(child_node)

                # If the child node is a type that allows clarification, insert the clarification node after it
                if isinstance(child_node, tuple(nodes_that_allow_clarification)):
                    clarification_node = LLMClarificationQuestionNode(f"clarification_{child_node.id}")
                    clarification_node.co = co

                    # Wrap the clarification node with a RepeatUntilNode
                    rep_till_fail_node = RepeatUntilNode(f"reptillfail_{clarification_node.id}")
                    rep_till_fail_node.children = [clarification_node]
                    rep_till_fail_node.co = co
                    print(f"Inserting Clarification Node after {child_node.id} - {rep_till_fail_node.repeat_condition}")

                    # Insert the RepeatUntilNode after the child node in the list of children
                    nodes.get(n).children.append(rep_till_fail_node)

                children = children["Next"]

        elif parser.bt_nodes[n]["Concept"] in ["RepeatUntilSuccess", "RepeatUntilFailure", "Limiter", "Repeater", "Inverter"]:
            children = parser.bt_nodes[n]["firstChild"]
            child_node = nodes.get(children["Id"])

            # Attach the original child node
            nodes.get(n).children = [child_node]

            # Insert Clarification Node if necessary, wrapped in a RepeatUntilNode
            if isinstance(child_node, tuple(nodes_that_allow_clarification)):
                clarification_node = LLMClarificationQuestionNode(f"clarification_{child_node.id}")
                clarification_node.co = co

                rep_till_fail_node = RepeatUntilNode(f"reptillfail_{clarification_node.id}")
                rep_till_fail_node.children = [clarification_node]
                rep_till_fail_node.co = co
                print(f"Inserting Clarification Node after {child_node.id} - {rep_till_fail_node.repeat_condition}")

                nodes.get(n).children.append(rep_till_fail_node)

        # Node-specific properties
        if parser.bt_nodes[n]["Concept"] in ["RepeatUntilSuccess", "RepeatUntilFailure", "Limiter", "Repeater"]:
            nodes.get(n).limit = parser.bt_nodes[n]["properties"]["maxLoop"]

        if parser.bt_nodes[n]["Concept"] in ["Question", "Need Question", "Persona Question", "Knowledge Question", "Confirm Question", "Target Question", "Target Type Question"]:
            nodes.get(n).question = parser.bt_nodes[n]["properties"]["question"]
            nodes.get(n).variable = parser.bt_nodes[n]["properties"]["variable"]

        if parser.bt_nodes[n]["Concept"] == "Greeter":
            nodes.get(n).variable = parser.bt_nodes[n]["properties"]["variable"]

        if parser.bt_nodes[n]["Concept"] == "Information":
            nodes.get(n).message = parser.bt_nodes[n]["properties"]["message"]

        # Handling modifiers
        if parser.bt_nodes[n]["Concept"] in ["World Modifier", "Usecase Modifier"]:
            if parser.bt_nodes[n]["properties"]:
                key = list(parser.bt_nodes[n]["properties"].keys())[0]
                nodes.get(n).variable = key
                val = parser.bt_nodes[n]["properties"][key]
                nodes.get(n).value = bool(val == "True")

        # Handling conditions
        if parser.bt_nodes[n]["Concept"] in ["Equal", "Condition"]:
            nodes.get(n).variables = {key: bool(val == "True") for key, val in parser.bt_nodes[n]["properties"].items()}
        
        if parser.bt_nodes[n]["Concept"] == "Equal Value":
            nodes.get(n).variables = {key: val for key, val in parser.bt_nodes[n]["properties"].items()}

        if parser.bt_nodes[n]["Concept"] == "Explanation Method":
            nodes.get(n).params = parser.bt_nodes[n]["params"] if "params" in parser.bt_nodes[n] else {}
            nodes.get(n).endpoint = parser.bt_nodes[n]["Instance"]

        if parser.bt_nodes[n]["Concept"] == "User Question":
            nodes.get(n).params = parser.bt_nodes[n]["params"] if "params" in parser.bt_nodes[n] else {}

    # Set the root node
    root_id = parser.bt_root
    root = node.RootNode('0')
    root.co = co
    root.children.append(nodes.get(root_id))

    return bt.Tree(root, nodes)

def printTree(root, level=0):
    print(" - " * level, root.toString())
    if hasattr(root, "children"):
        for child in root.children:
            printTree(child, level + 1)