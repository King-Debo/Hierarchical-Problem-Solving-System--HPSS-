# This is the file that contains the utility functions that are used by the other files, such as data processing, conversion, optimization, evaluation, etc.

# Import the necessary modules
import torch
import numpy as np
import requests
import json

# Define the function that samples a batch of transitions from a memory
def sample(memory, batch_size):
    # Randomly select a batch of indices from the memory
    indices = np.random.choice(len(memory), batch_size, replace=False)
    # Initialize empty lists for the states, the meta actions, the actions, the rewards, the next states, and the dones
    states = []
    meta_actions = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    # For each index in the batch, append the corresponding transition to the lists
    for i in indices:
        state, meta_action, action, reward, next_state, done = memory[i]
        states.append(state)
        meta_actions.append(meta_action)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
    # Convert the lists to tensors
    states = torch.stack(states)
    meta_actions = torch.stack(meta_actions)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    next_states = torch.stack(next_states)
    dones = torch.stack(dones)
    # Return the batch of transitions
    return states, meta_actions, actions, rewards, next_states, dones

# Define the function that performs the soft update of the target parameters with the source parameters
def soft_update(target, source, tau):
    # For each parameter in the target and the source, update the target parameter with a weighted average of the target and the source parameters, using the soft update rate
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

# Define the function that converts the input to natural language, using an online API
def convert_to_natural_language(input):
    # Define the URL of the API
    url = "https://api.deepai.org/api/text-generator"
    # Define the headers of the API
    headers = {"api-key": "sk-7f6a4f6a4f6a4f6a4f6a4f6a"}
    # Define the data of the API
    data = {"text": input}
    # Make a post request to the API
    response = requests.post(url, headers=headers, data=data)
    # Parse the response as a JSON object
    response = json.loads(response.text)
    # Extract the natural language output from the response
    natural_language_output = response["output"]
    # Return the natural language output
    return natural_language_output

# Define the function that converts the output from natural language to the desired format, using an online API
def convert_from_natural_language(natural_language_output):
    # Define the URL of the API
    url = "https://api.deepai.org/api/text2img"
    # Define the headers of the API
    headers = {"api-key": "sk-7f6a4f6a4f6a4f6a4f6a4f6a"}
    # Define the data of the API
    data = {"text": natural_language_output}
    # Make a post request to the API
    response = requests.post(url, headers=headers, data=data)
    # Parse the response as a JSON object
    response = json.loads(response.text)
    # Extract the output from the response
    output = response["output_url"]
    # Return the output
    return output
