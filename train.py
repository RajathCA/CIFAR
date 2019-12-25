#from models.feedforward import feedforward
from data_loader import DataLoader

# Define placeholders (inputs and outputs given to model)





# Define model (graph)
# Start with feedforward model













# Define Cross-Entropy Loss




# Define optimizer
# and the optimize operation




# Define session and initialize weights




# For training, implement the following using DataLoader Object
# Run training MAX_EPOCHS times
# In every epoch, cycle through all training data randomly
# After every epoch, save model weights in folder model_weights using tf.train.Saver()
# After every epoch, compute avg. loss on validation set
# Stop training once validation cost starts increasing
# Split entire training data into train (90%) and val (10%) sets

MAX_EPOCHS = 15
















# Load weights using tf.train.Saver() from epoch with lowest validation cost.
# Compute avg. test accuracy and loss






