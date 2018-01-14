from __future__ import print_function
import tensorflow as tf


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(3.2)
print(node1, node2)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log', sess.graph)
    val_node1_node_2 = sess.run([node1, node2])
    print(val_node1_node_2)

    node3 = tf.add(node1, node2)
    print('node3:', node3)
    print('sess.run(node3): ', sess.run(node3))
writer.close()
