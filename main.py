# This is the main file that runs the system, and interacts with the user and the environment.
# It contains the main loop that takes the input from the user, passes it to the hierarchical system, and returns the output to the user.
# It also handles the exceptions and errors that may occur during the execution.

# Import the necessary modules
import hierarchical_system as hs
import utils as ut

# Create an instance of the hierarchical system
system = hs.HierarchicalSystem()

# Define a flag to indicate whether the system is running or not
running = True

# Define a welcome message to greet the user
welcome_message = "Hello, this is a hierarchical system that can decompose and compose complex problems and solutions using multiple levels of abstraction. I can help you with various tasks, such as natural language generation, program synthesis, and automated reasoning. Please enter your problem statement, query, command, or feedback in natural language, and I will try to respond accordingly. You can also enter 'quit' to exit the system."

# Print the welcome message
print(welcome_message)

# Start the main loop
while running:
    # Try to get the input from the user
    try:
        user_input = input("User: ")
    # If there is an error in getting the input, print an error message and continue the loop
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    # If the user enters 'quit', set the flag to False and break the loop
    if user_input.lower() == 'quit':
        running = False
        break
    
    # Try to pass the input to the hierarchical system and get the output
    try:
        system_output = system.process(user_input)
    # If there is an error in processing the input, print an error message and continue the loop
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    # Print the output from the system
    print(f"System: {system_output}")

# Define a goodbye message to thank the user
goodbye_message = "Thank you for using the hierarchical system. I hope you enjoyed the interaction and found it useful. Have a nice day!"

# Print the goodbye message
print(goodbye_message)
