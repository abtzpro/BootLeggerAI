# BootLeggerAI
A python based chat bot for coders

BootLegger AI is a Python script that uses deep learning, neural networking, and natural language processing to create a chatbot that can help users with their programming needs. The chatbot can understand natural language and generate appropriate responses based on the user's input.

## Instructions for Use

1. Install the necessary libraries: Tensorflow and Numpy. You can do this using pip or any package manager of your choice. 

2. Copy the code in the BootLegger2.0.py file into your Python editor of choice.

3. Run the code in your Python environment. 

4. The chatbot will start running and prompt the user to enter their request.

5. Enter your request, and the chatbot will generate a response based on the predicted output.

6. If you encounter any issues, please see the Troubleshooting section below.

## Detailed instructions

1. Install Required Libraries: Before running the script, make sure you have installed all the required libraries. This script requires `tensorflow` and `numpy`. You can install them via pip or conda:

```
pip install tensorflow numpy
```

2. Prepare Input and Output Data: The script takes in an array of input data and an array of output data. Each element of the input array should be a string that represents a programming task you want help with. The output array should contain the corresponding output for each input task. 

For example, you can create a numpy array for the input and output data like this:

```
import numpy as np

# Define input and output data
input_data = np.array(['create a python script', 'build a program', 'generate a code'])
output_data = np.array([['create', 'python', 'script'], ['build', 'program'], ['generate', 'code']])
```

3. Tokenize Input Data: To use the input data with the model, we need to tokenize it first. Tokenization is the process of converting text into numerical values. The function `tokenize_input` in the script takes in the input data and returns the tokenizer object, the tokenized input sequence, the maximum length of the input sequence, and the vocabulary size. 

You can tokenize the input data like this:

```
from bootlegger_ai import tokenize_input

tokenizer, input_seq, max_len, vocab_size = tokenize_input(input_data)
```

4. Define the Neural Network Model: The next step is to define the neural network model. The function `define_model` in the script takes in the vocabulary size and maximum length of the input sequence and returns the model object. 

You can define the model like this:

```
from bootlegger_ai import define_model

model = define_model(vocab_size, max_len)
```

5. Train the Neural Network Model: After defining the model, we need to train it with the input and output data. The function `train_model` in the script takes in the model object, input sequence, output data, and number of epochs to train the model. It returns the trained model object.

You can train the model like this:

```
from bootlegger_ai import train_model

model = train_model(model, input_seq, output_data)
```

6. Test the Model: After training the model, we can test it on new input data. The function `test_model` in the script takes in the model object, test data, tokenizer object, and maximum length of the input sequence. It returns the predictions for the test data.

You can test the model like this:

```
from bootlegger_ai import test_model

test_data = np.array(['I want to create a new website'])
predictions = test_model(model, test_data, tokenizer, max_len)
```

7. Generate Response: Finally, we can generate a response based on the predicted output. The function `generate_response` in the script takes in the predictions and tokenizer object and returns a response string.

You can generate a response like this:

```
from bootlegger_ai import generate_response

response = generate_response(predictions, tokenizer)
print(response)
```

And that's it! By following these steps, you can use the BootLegger AI script to generate responses to programming-related requests.

## Developed By

This script was developed by Adam Rivers and Hello Security LLC.

## Troubleshooting

If the chatbot is not generating appropriate responses, please ensure that the input data is relevant to the context of programming. 

Additionally, you can try retraining the neural network model by modifying the input and output data in the script.

If you encounter any other issues, please feel free to reach out for assistance.
