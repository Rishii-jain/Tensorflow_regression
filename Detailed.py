# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Disable eager execution for TensorFlow 2.x compatibility
tf.compat.v1.disable_eager_execution()

# Generate random data for x and y
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

# Plot the random data points
plt.plot(x_data, y_label)

# Define the variables for slope (m) and intercept (b)
m = tf.Variable(0.44)  # slope
b = tf.Variable(0.87)   # constant

# Define the error (loss) function
error = 0

# Calculate the total error using mean squared error
for x, y in zip(x_data, y_label):
    y_hat = m * x + b  # predicted y value
    error += (y - y_hat) ** 2  # mean squared error

# Define the optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)  # minimize the error function

# Initialize variables
init = tf.compat.v1.global_variables_initializer()

# Start a TensorFlow session
with tf.compat.v1.Session() as sess:
    sess.run(init)  # Initialize variables
    
    training_steps = 1000  # Number of training steps
    
    # Perform training steps to optimize the variables (slope and intercept)
    for i in range(training_steps):
        sess.run(train)  # Execute one training step
        
    # Get the final values of slope and intercept after training
    final_slope, final_intercept = sess.run([m, b])

# Generate test data for plotting the regression line
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept  # Predicted y values using the final slope and intercept

# Plot the regression line along with the original data points
plt.plot(x_test, y_pred_plot, 'r')  # Plot the regression line in red
plt.plot(x_data, y_label, '*')  # Plot the original data points as asterisks
plt.show()  # Show the plot
