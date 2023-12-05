# This is the file that implements the hierarchical neural network, and encodes and decodes the hierarchical representation of the problem or the solution.
# It contains the class that defines the hierarchical neural network, and its methods that forward and backward the information through the network.

# Import the necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the class for the hierarchical neural network
class HierarchicalNeuralNetwork(nn.Module):
    # Define the constructor for the class
    def __init__(self, input_size, output_size, reward_size):
        # Call the constructor of the parent class
        super(HierarchicalNeuralNetwork, self).__init__()
        # Initialize the input size, the output size, and the reward size
        self.input_size = input_size
        self.output_size = output_size
        self.reward_size = reward_size
        # Initialize the hidden size
        self.hidden_size = 256
        # Initialize the embedding layer, using an embedding matrix
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Initialize the encoder layer, using a recurrent neural network
        self.encoder = nn.RNN(hidden_size, hidden_size, batch_first=True)
        # Initialize the decoder layer, using a recurrent neural network
        self.decoder = nn.RNN(hidden_size, hidden_size, batch_first=True)
        # Initialize the output layer, using a linear layer
        self.output = nn.Linear(hidden_size, output_size)
        # Initialize the reward layer, using a linear layer
        self.reward = nn.Linear(hidden_size, reward_size)
    
    # Define the method that forwards the information through the network
    def forward(self, input):
        # Embed the input into a hidden representation
        embedded = self.embedding(input)
        # Encode the embedded input into a hidden state
        _, hidden = self.encoder(embedded)
        # Decode the hidden state into an output representation
        output, _ = self.decoder(hidden)
        # Apply the output layer to the output representation
        output = self.output(output)
        # Apply the softmax function to the output
        output = F.softmax(output, dim=-1)
        # Apply the reward layer to the output representation
        reward = self.reward(output)
        # Return the output and the reward
        return output, reward
