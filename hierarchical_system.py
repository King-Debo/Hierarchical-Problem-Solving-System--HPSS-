# This is the file that implements the hierarchical system, and encodes and decodes the problem and the solution using multiple levels of abstraction.
# It contains the class that defines the hierarchical system, and its methods that generate and parse the hierarchical representations, using the large language model.

# Import the necessary modules
import torch
import transformers
import utils as ut

# Define the class for the hierarchical system
class HierarchicalSystem:
    # Define the constructor for the class
    def __init__(self):
        # Initialize the large language model, such as GPT-3
        self.model = transformers.AutoModelForCausalLM.from_pretrained("gpt3-large")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt3-large")
    
    # Define the method that processes the input from the user and returns the output to the user
    def process(self, user_input):
        # Convert the user input to natural language
        user_input = ut.convert_to_natural_language(user_input)
        # Generate the hierarchical representation of the problem or the solution, using the large language model
        hierarchical_representation = self.generate_hierarchical_representation(user_input)
        # Parse the hierarchical representation of the problem or the solution, using the large language model
        natural_language_output = self.parse_hierarchical_representation(hierarchical_representation)
        # Convert the natural language output to the desired format
        system_output = ut.convert_from_natural_language(natural_language_output)
        # Return the system output
        return system_output
    
    # Define the method that generates the hierarchical representation of the problem or the solution, using the large language model
    def generate_hierarchical_representation(self, natural_language_input):
        # Define the prefix for the generation task
        prefix = "Generate a hierarchical representation of the following problem or solution in natural language:\n"
        # Define the suffix for the generation task
        suffix = "\nEnd of hierarchical representation."
        # Concatenate the prefix, the natural language input, and the suffix
        input_text = prefix + natural_language_input + suffix
        # Encode the input text into tokens
        input_tokens = self.tokenizer.encode(input_text, return_tensors="pt")
        # Generate the output tokens using the large language model
        output_tokens = self.model.generate(input_tokens, max_length=1024, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)
        # Decode the output tokens into text
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # Extract the hierarchical representation from the output text
        hierarchical_representation = output_text.split(suffix)[0].strip()
        # Return the hierarchical representation
        return hierarchical_representation
    
    # Define the method that parses the hierarchical representation of the problem or the solution, using the large language model
    def parse_hierarchical_representation(self, hierarchical_representation):
        # Define the prefix for the parsing task
        prefix = "Parse the following hierarchical representation of the problem or the solution in natural language:\n"
        # Define the suffix for the parsing task
        suffix = "\nEnd of natural language output."
        # Concatenate the prefix, the hierarchical representation, and the suffix
        input_text = prefix + hierarchical_representation + suffix
        # Encode the input text into tokens
        input_tokens = self.tokenizer.encode(input_text, return_tensors="pt")
        # Generate the output tokens using the large language model
        output_tokens = self.model.generate(input_tokens, max_length=1024, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)
        # Decode the output tokens into text
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # Extract the natural language output from the output text
        natural_language_output = output_text.split(suffix)[0].strip()
        # Return the natural language output
        return natural_language_output
