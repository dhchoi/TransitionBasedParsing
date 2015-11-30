import sys
import tensorflow as tf
import Constants as C
from Transition import Transition


class DeepLearningModel:

    def __init__(self, labeled, posTypes, labelTypes):
        self.labeled = labeled
        self.posTypes = posTypes
        self.labelTypes = labelTypes
        self.numFeatures = 7
        self.numLabels = 3

        self.X = []
        self.Y = []
        self.hasLearned = False

        # TensorFlow Initializations
        self.x = tf.placeholder("float", [None, self.numFeatures])
        self.y_ = tf.placeholder("float", [None, self.numLabels])

        W = tf.Variable(tf.zeros([self.numFeatures, self.numLabels]))
        b = tf.Variable(tf.zeros([self.numLabels]))
        self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)

        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

        self.getBestLabel = tf.argmax(self.y, 1)

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

        return featureVector

    def createLabelVector(self, correct_transition):
        labelVector = [0] * self.numLabels
        labelVector[correct_transition.transitionType] = 1

        assert len(labelVector) == self.numLabels

        return labelVector

    def learn(self, correct_transition, stack, buff, labels, previous_transitions):
        # self.sess.run(self.train_step,
        #               feed_dict={self.x: [self.createFeatureVector(stack, buff, labels, previous_transitions)],
        #                          self.y_: [self.createLabelVector(correct_transition)]})
        self.X.append(self.createFeatureVector(stack, buff, labels, previous_transitions))
        self.Y.append(self.createLabelVector(correct_transition))

    def predict(self, stack, buff, labels, previous_transitions):
        if not self.hasLearned:
            print >> sys.stderr, "len(X): " + str(len(self.X))
            print >> sys.stderr, "len(Y): " + str(len(self.Y))
            self.sess.run(self.train_step, feed_dict={self.x: self.X, self.y_: self.Y})
            self.hasLearned = True

        bestTransitionType = self.sess.run(self.getBestLabel,
                                           feed_dict={self.x: [self.createFeatureVector(stack, buff, labels, previous_transitions)]})

        print >> sys.stderr, 'bestTransitionType: %d' % bestTransitionType[0]

        if len(stack) < 2 and bestTransitionType[0] != 0:
            return (0, Transition(0, None))

        if len(buff) == 0 and bestTransitionType[0] == 0:
            # print >> sys.stderr, 'changing shift to sth else'
            # return (0, Transition(1, None)) if probs[1] > probs[2] else (0, Transition(2, None))
            return (0, Transition(1, None))

        return (0, Transition(bestTransitionType[0], None))
