#simple regression example
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

plt.plot(x_data,y_label)

# y = mx+b
# np.random.rand(2)

m = tf.Variable(0.44) #slope
b = tf.Variable(0.87) #constant

error  = 0

for x,y in zip(x_data, y_label):
    y_hat = m*x + b
    error += (y-y_hat)**2
    
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    
    training_steps = 1000
    
    for i in range(training_steps):
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m,b])
    
x_test = np.linspace(-1,11,10)

y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot, 'r')  
plt.plot(x_data,y_label,'*')
plt.show()      
    