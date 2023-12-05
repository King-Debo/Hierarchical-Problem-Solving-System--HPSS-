# This is the file that implements the hierarchical learning, and learns from both bottom-up and top-down information.
# It contains the classes and functions that implement and use various components and techniques, such as hierarchical reinforcement learning, hierarchical Bayesian networks, hierarchical neural networks, and hierarchical planning, to learn from the data and the environment.

# Import the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils as ut

# Define the class for the hierarchical reinforcement learning
class HierarchicalReinforcementLearning:
    # Define the constructor for the class
    def __init__(self, state_size, action_size, meta_action_size, reward_size, gamma, alpha, beta, epsilon, tau):
        # Initialize the state size, the action size, the meta action size, the reward size, the discount factor, the learning rate, the entropy coefficient, the exploration rate, and the soft update rate
        self.state_size = state_size
        self.action_size = action_size
        self.meta_action_size = meta_action_size
        self.reward_size = reward_size
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tau = tau
        # Initialize the meta controller, the meta target, the controller, and the target, using hierarchical neural networks
        self.meta_controller = HierarchicalNeuralNetwork(state_size, meta_action_size, reward_size)
        self.meta_target = HierarchicalNeuralNetwork(state_size, meta_action_size, reward_size)
        self.controller = HierarchicalNeuralNetwork(state_size + meta_action_size, action_size, reward_size)
        self.target = HierarchicalNeuralNetwork(state_size + meta_action_size, action_size, reward_size)
        # Initialize the meta optimizer and the optimizer, using Adam
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=alpha)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=alpha)
        # Initialize the meta memory and the memory, using lists
        self.meta_memory = []
        self.memory = []
    
    # Define the method that selects a meta action using the meta controller and the epsilon-greedy policy
    def select_meta_action(self, state):
        # Convert the state to a tensor
        state = torch.tensor(state, dtype=torch.float32)
        # Generate a random number
        rand = np.random.random()
        # If the random number is less than the epsilon, select a random meta action
        if rand < self.epsilon:
            meta_action = np.random.randint(self.meta_action_size)
        # Otherwise, select the meta action with the highest value, using the meta controller
        else:
            meta_action = self.meta_controller(state).argmax().item()
        # Return the meta action
        return meta_action
    
    # Define the method that selects an action using the controller and the epsilon-greedy policy
    def select_action(self, state, meta_action):
        # Convert the state and the meta action to tensors
        state = torch.tensor(state, dtype=torch.float32)
        meta_action = torch.tensor(meta_action, dtype=torch.float32)
        # Concatenate the state and the meta action
        state_meta_action = torch.cat((state, meta_action))
        # Generate a random number
        rand = np.random.random()
        # If the random number is less than the epsilon, select a random action
        if rand < self.epsilon:
            action = np.random.randint(self.action_size)
        # Otherwise, select the action with the highest value, using the controller
        else:
            action = self.controller(state_meta_action).argmax().item()
        # Return the action
        return action
    
    # Define the method that stores the transition in the meta memory or the memory
    def store(self, state, meta_action, action, reward, next_state, done, meta):
        # Convert the state, the meta action, the action, the reward, the next state, and the done to tensors
        state = torch.tensor(state, dtype=torch.float32)
        meta_action = torch.tensor(meta_action, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        # If the meta flag is True, store the transition in the meta memory
        if meta:
            self.meta_memory.append((state, meta_action, reward, next_state, done))
        # Otherwise, store the transition in the memory
        else:
            self.memory.append((state, meta_action, action, reward, next_state, done))
    
    # Define the method that updates the meta controller and the controller using the meta memory and the memory
    def update(self, batch_size):
        # If the meta memory and the memory are not empty, sample a batch of transitions from them
        if self.meta_memory and self.memory:
            meta_batch = ut.sample(self.meta_memory, batch_size)
            batch = ut.sample(self.memory, batch_size)
            # For each transition in the meta batch, unpack the state, the meta action, the reward, the next state, and the done
            for state, meta_action, reward, next_state, done in meta_batch:
                # Calculate the target value, using the meta target and the reward
                target_value = reward + self.gamma * self.meta_target(next_state).max() * (1 - done)
                # Calculate the expected value, using the meta controller and the meta action
                expected_value = self.meta_controller(state).gather(0, meta_action)
                # Calculate the meta loss, using the mean squared error
                meta_loss = nn.MSELoss()(expected_value, target_value)
                # Calculate the entropy, using the meta controller and the meta action
                entropy = -self.beta * self.meta_controller(state).log().gather(0, meta_action)
                # Calculate the total loss, using the meta loss and the entropy
                total_loss = meta_loss - entropy
                # Zero the gradients of the meta optimizer
                self.meta_optimizer.zero_grad()
                # Backpropagate the total loss
                total_loss.backward()
                # Update the parameters of the meta controller
                self.meta_optimizer.step()
                # Update the parameters of the meta target, using the soft update
                ut.soft_update(self.meta_target, self.meta_controller, self.tau)
            # For each transition in the batch, unpack the state, the meta action, the action, the reward, the next state, and the done
            for state, meta_action, action, reward, next_state, done in batch:
                # Concatenate the state and the meta action
                state_meta_action = torch.cat((state, meta_action))
                # Concatenate the next state and the meta action
                next_state_meta_action = torch.cat((next_state, meta_action))
                # Calculate the target value, using the target and the reward
                target_value = reward + self.gamma * self.target(next_state_meta_action).max() * (1 - done)
                # Calculate the expected value, using the controller and the action
                expected_value = self.controller(state_meta_action).gather(0, action)
                # Calculate the loss, using the mean squared error
                loss = nn.MSELoss()(expected_value, target_value)
                # Zero the gradients of the optimizer
                self.optimizer.zero_grad()
                # Backpropagate the loss
                loss.backward()
                # Update the parameters of the controller
                self.optimizer.step()
                # Update the parameters of the target, using the soft update
                ut.soft_update(self.target, self.controller, self.tau)
