import tensorflow as tf

def conv_layer(input, size_in, size_out, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=[5, 5, size_in, size_out], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='B')
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fc_layer(input, size_in, size_out, name='fc'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='B')
        act = tf.matmul(input, w) + b
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return act

class Mnist():
    def __init__(self, learning_rate, logdir='./tmp/'):
        self.learning_rate = learning_rate
        self.mnist = tf.contrib.learn.datasets.mnist.read_data_sets(
            train_dir=logdir + "data",
            one_hot=True)
        self.graph = self.build()
        self.logdir = logdir
        
    def build(self):

        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 3)
        y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

        conv0 = conv_layer(x_image, 1, 32, 'conv0')
        conv1 = conv_layer(conv0, 32, 64, 'conv1')

        flattened = tf.reshape(conv1, [-1, 7 * 7 * 64])

        fc0 = fc_layer(flattened, 7 * 7 * 64, 1024, 'fc0')
        relu = tf.nn.relu(fc0)
        logits = fc_layer(relu, 1024, 10, 'fc1')

        with tf.name_scope('xent'):
            xent = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=y),
                name='xent'
            )
            tf.summary.scalar('xent', xent)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(xent)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1),
                tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        summ = tf.summary.merge_all()

        return {'input' : x,
                'logits' : logits,
                'y' : y,
                'train_step' : train_step,
                'accuracy' : accuracy,
                'summ' : summ}

    def train(self, nb_epochs, model_path=None):

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if model_path:
                saver.restore(sess, model_path)

            writer = tf.summary.FileWriter(self.logdir)
            writer.add_graph(sess.graph)

            for i in range(nb_epochs):
                batch = self.mnist.train.next_batch(100)
                if i % 5 == 0:
                    train_accuracy, s = sess.run([self.graph['accuracy'], self.graph['summ']],
                                                 feed_dict={self.graph['input']: batch[0],
                                                            self.graph['y']: batch[1]})
                    print('epoch : {} - accuracy : {}'.format(i, train_accuracy))
                    writer.add_summary(s, i)
                if i % 500 == 0:
                    saver.save(sess, self.logdir+'model/model.ckpt', i)

                sess.run(self.graph['train_step'], feed_dict={
                    self.graph['input'] : batch[0],
                    self.graph['y'] : batch[1]})

def main():
    nn = Mnist(learning_rate=1E-3)
    nn.train(nb_epochs=1000)

if __name__ == '__main__':
    main()