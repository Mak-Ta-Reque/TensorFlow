import tensorflow as tf


W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x + b

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

print('Output  y  without learning :', sess.run(linear_model, {x: [1, 2, 3, 4, 5, 6]}))


y = tf.placeholder(tf.float32)
square_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(square_deltas)
print('Error of the linear model: ', sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

print('Considering W = -1 and b = 1  minimizes loss to zero')

fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])
sess.run([fixW,fixb])

print('Loss after fixing W and b: ', sess.run(loss,{x: [1, 2, 3, 4], y : [0, -1, -2, -3]}))


print('Now linear model will be optimizer by tf.train API')

learningrate = 0.01;
optimizer = tf.train.GradientDescentOptimizer(learningrate)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W,b]))
