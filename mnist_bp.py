#encoding='utf-8'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#mnist数据集相关常数
INPUT_NODE = 784    #图片像素
OUTPUT_NODE = 10    #0-9十个数字

#配置神经网络参数
LAYER1_NODE = 500   #隐藏层节点数500
BATCH_SIZE  = 100   #一个训练batch中数据个数
LEARNING_RATE_BASE = 0.8           #基础学习率
LEARNING_RATE_DECAY = 0.99    #学习率衰减率
REGULARIZATION_RATE = 0.0001  #描述模型复杂度的正则项在损失函数中的系数 
TRAINING_STEPS = 5000        #训练轮数
MOVING_AVERAGE_DECAY = 0.99   #滑动平均衰减率

#计算神经网络前向传播结果
def inference(input_tensor, avg_class, weights1,biases1,weights2, biases2):
    if avg_class == None:   #没有提供划动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+
        avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

#训练模型过程
def train(mnist):
    x = tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name = 'x-input')
    y_= tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name= 'y-input')

    #隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算当前参数下前向传播结果
    y = inference(x,None,weights1,biases1,weights2,biases2)

    #定义存储训练轮数变量
    global_step = tf.Variable(0,trainable=False)

    #初始化划动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #在所有代表神经网络参数的变量上使用划动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算使用划动平均之后的前向传播结果
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    #损失函数：预测值和真实值的交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数 
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失
    regularization = regularizer(weights1)+regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization
    #设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
    global_step,
    mnist.train.num_examples/BATCH_SIZE,
    LEARNING_RATE_DECAY,
    staircase=True)     

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #反向传播更新参数及参数的划动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    #计算预测答案
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



    #初始化并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x: mnist.test.images,y_:mnist.test.labels}

        #迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i%1000== 0:
                validate_acc = sess.run(accuracy,feed_dict = validate_feed)
                print('After %d training steps, validation accuracy using average model is %g' %(i,validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g" %(TRAINING_STEPS,test_acc))            
def main(argv = None):
    #下载mnist数据
    mnist = input_data.read_data_sets("./to/MNIST_data/",one_hot=True)
    train(mnist)
if __name__ == '__main__':
    main()