import sys
import tensorflow as tf
import Constants as C
import numpy as np
from sklearn import svm, datasets
from Transition import Transition


class MyModel:

    def __init__(self, labeled, posTypes, labelTypes):
        self.labeled = labeled
        self.posTypes = posTypes
        self.labelTypes = labelTypes
        self.numFeatures = 7
        self.numLabels = 3

        # # TensorFlow Initializations
        # self.x = tf.placeholder("float", [None, self.numFeatures])
        # self.y_ = tf.placeholder("float", [None, self.numLabels])
        #
        # W = tf.Variable(tf.zeros([self.numFeatures, self.numLabels]))
        # b = tf.Variable(tf.zeros([self.numLabels]))
        # self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)
        #
        # cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        # self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        #
        # init = tf.initialize_all_variables()
        # self.sess = tf.Session()
        # self.sess.run(init)
        #
        # self.getBestLabel = tf.argmax(self.y, 1)

        # SciKit
        self.X = []
        self.Y = []
        self.clf = svm.SVC(probability=False)
        self.hasFitted = False

    def createFeatureVector(self, stack, buff, labels, previous_transitions):
        # featureVector: [stack[-1].pos, stack[-2].pos,
        #                 buff[-1].pos, buff[-2].pos, buff[-3].pos, buff[-4].pos
        #                 previous_transitions[-1].transitionType]
        featureVector = []

        # Top two POS tags from the stack
        for i in range(2):
            if i >= len(stack):
                featureVector.append(-1)
            else:
                w = stack[-(i + 1)]
                pos = w[C.POSTAG]
                featureVector.append(self.posTypes.index(pos) if pos in self.posTypes else -1)

        # Next four POS tags from the buffer
        for i in range(4):
            if i >= len(buff):
                featureVector.append(-1)
            else:
                w = buff[-(i + 1)]
                pos = w[C.POSTAG]
                featureVector.append(self.posTypes.index(pos) if pos in self.posTypes else -1)

        # Previous transition type
        if len(previous_transitions) > 0:
            featureVector.append(previous_transitions[-1].transitionType)
        else:
            featureVector.append(-1)

        # if self.labeled:
            # # Action and label pair
            # features['transition=%d,label=%s' % (tType, tLabel)] = 1
            # # Label bias
            # features['label=%s' % (tLabel)] = 1

        assert len(featureVector) == self.numFeatures

        # print >> sys.stderr, tf.transpose(featureVector)

        return featureVector

    def createLabelVector(self, correct_transition):
        labelVector = [0] * self.numLabels
        labelVector[correct_transition.transitionType] = 1

        assert len(labelVector) == self.numLabels

        # print >> sys.stderr, labelVector

        return labelVector  # labelVector

    def learn(self, correct_transition, stack, buff, labels, previous_transitions):
        # self.sess.run(self.train_step,
        #               feed_dict={self.x: tf.zeros([self.numFeatures]), # self.createFeatureVector(stack, buff, labels, previous_transitions),
        #                          self.y_: tf.zeros([self.numLabels]) }) # self.createLabelVector(correct_transition)})
        self.X.append(self.createFeatureVector(stack, buff, labels, previous_transitions))
        # print >> sys.stderr, "correctTransitionType: " + str(correct_transition.transitionType)
        self.Y.append(correct_transition.transitionType)

    def predict(self, stack, buff, labels, previous_transitions):
        if not self.hasFitted:
            print >> sys.stderr, "Start fitting model"
            print >> sys.stderr, "len(X): " + str(len(self.X))
            print >> sys.stderr, "len(Y): " + str(len(self.Y))
            self.clf.fit(self.X, self.Y)
            print >> sys.stderr, "Finished fitting model"
            self.hasFitted = True

        # bestTransitionType = self.sess.run(self.getBestLabel,
        #                                    feed_dict={self.x: self.createFeatureVector(stack, buff, labels, previous_transitions)})
        bestTransitionType = self.clf.predict([self.createFeatureVector(stack, buff, labels, previous_transitions)])
        # print >> sys.stderr, 'bestTransitionType: %d' % bestTransitionType[0]

        if len(stack) < 2 and bestTransitionType[0] != 0:
            return (0, Transition(0, None))

        if len(buff) == 0 and bestTransitionType[0] == 0:
            # print >> sys.stderr, 'changing shift to sth else'
            # probs = self.clf.predict_proba([self.createFeatureVector(stack, buff, labels, previous_transitions)])[0]
            # return (0, Transition(1, None)) if probs[1] > probs[2] else (0, Transition(2, None))
            return (0, Transition(1, None))

        return (0, Transition(bestTransitionType[0], None))


def runTensorSample():
    # 1797 data
    mnistData = datasets.load_digits()
    images = mnistData.images.reshape((len(mnistData.images), -1))
    labels = []

    for label in mnistData.target:
        l = [0] * 10
        l[label] = 1
        labels.append(l)

    x = tf.placeholder("float", [None, 64])
    W = tf.Variable(tf.zeros([64, 10]))
    b = tf.Variable(tf.zeros([10]))

    y_ = tf.placeholder("float", [None, 10])
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # for i in range(100):
    #     batch_xs, batch_ys = mnist.train.next_batch(100)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print images
    print labels

    sess.run(train_step, feed_dict={x: images, y_: labels})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print sess.run(accuracy, feed_dict={x: images, y_: labels})

def runSvmSample():
    X = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
    y = [0, 1, 2]
    # clf = svm.SVC(probability=True)
    clf = svm.LinearSVC()
    clf.fit(X, y)
    X_ = [0, 0, 1]
    # print clf.predict_proba([X_])
    # print clf.predict([X_])
    print clf.predict([X_])
    result = clf.decision_function([X_])[0]
    print result

    votes = np.zeros(len(y))
    p = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if result[p] > 0:
                votes[i] += 1
            else:
                votes[i] += 1
            p += 1
    print votes

# runSvmSample()