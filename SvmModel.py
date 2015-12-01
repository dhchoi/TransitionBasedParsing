import sys
import Constants as C
import numpy
import scipy
from sklearn import svm, datasets
from Transition import Transition


class SvmModel:

    def __init__(self, labeled, posTypes, labelTypes):
        self.labeled = labeled
        self.posTypes = posTypes
        self.labelTypes = labelTypes

        # SciKit
        self.featureDict = {}
        self.features = []
        self.labels = []
        self.clf = svm.SVC(
                kernel='poly',
                degree=2,
                coef0=0,
                gamma=0.2,
                C=0.5,
                verbose=True,
                probability=True)
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

        # assert len(featureVector) == self.numFeatures

        # print >> sys.stderr, tf.transpose(featureVector)

        return featureVector

    def createLabelVector(self, correct_transition):
        labelVector = [0] * self.numLabels
        labelVector[correct_transition.transitionType] = 1

        # assert len(labelVector) == self.numLabels

        # print >> sys.stderr, labelVector

        return labelVector  # labelVector

    # def possibleTransitions(self, stack, buff):
    #     possible_transitions = []
    #     # if len(buff) >= 1:
    #     #     possible_transitions.append(Transition(Transition.Shift, None))
    #     # if len(stack) >= 2:
    #     #     for label in self.label_set:
    #     #         possible_transitions.append(Transition(Transition.LeftArc, label))
    #     #         possible_transitions.append(Transition(Transition.RightArc, label))
    #     # assert len(possible_transitions) > 0
    #     return possible_transitions

    def learn(self, correct_transition, stack, buff, labels, previous_transitions):
        self.features.append(self.createFeatureVector(stack, buff, labels, previous_transitions))
        # print >> sys.stderr, "correctTransitionType: " + str(correct_transition.transitionType)
        self.labels.append(correct_transition.transitionType)

    def predict(self, stack, buff, labels, previous_transitions):
        if not self.hasFitted:
            print >> sys.stderr, "Start fitting model"
            print >> sys.stderr, "len(features): " + str(len(self.features))
            print >> sys.stderr, "len(labels): " + str(len(self.labels))
            self.clf.fit(self.features, self.labels)
            print >> sys.stderr, "Finished fitting model"
            self.hasFitted = True

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


# def runSvmSample():
#     X = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
#     y = [0, 1, 2]
#     # clf = svm.SVC(probability=True)
#     clf = svm.LinearSVC()
#     clf.fit(X, y)
#     X_ = [0, 0, 1]
#     # print clf.predict_proba([X_])
#     # print clf.predict([X_])
#     print clf.predict([X_])
#     result = clf.decision_function([X_])[0]
#     print result
#
#     votes = np.zeros(len(y))
#     p = 0
#     for i in range(len(y)):
#         for j in range(i + 1, len(y)):
#             print i, j, p
#             if result[p] > 0:
#                 votes[i] += 1
#             else:
#                 votes[j] += 1
#             p += 1
#     print votes
#
# runSvmSample()


_dictionary = {} # {'STK_4_POS_xxx': 4, 'STK_5_POS_xxx': 5, 'STK_1_POS_xxx': 1, 'STK_2_POS_xxx': 2, 'STK_3_POS_xxx': 3, 'STK_0_POS_xxx': 0}

def _convert_to_binary_features(features):
    """
    :param features: list of feature string which is needed to convert to binary features
    :type features: list(str)
    :return : string of binary features in libsvm format  which is 'featureID:value' pairs
    """
    unsorted_result = []
    for feature in features:
        _dictionary.setdefault(feature, len(_dictionary))
        unsorted_result.append(_dictionary[feature])

    # Default value of each feature is 1.0
    return ' '.join(str(featureID) + ':1.0' for featureID in sorted(unsorted_result))

features = ['STK_0_POS_xxx', 'STK_1_POS_xxx', 'STK_2_POS_xxx', 'STK_3_POS_xxx'] # extract_features
binary_features = _convert_to_binary_features(features) # "0:1.0 1:1.0 2:1.0 3:1.0"   =>   binary_features
features = ['STK_2_POS_xxx', 'STK_3_POS_xxx', 'STK_4_POS_xxx', 'STK_5_POS_xxx']
binary_features = _convert_to_binary_features(features) # 2:1.0 3:1.0 4:1.0 5:1.0

# gives out X, Y as
# (X, Y) = (binary_features, transition)
# training_seq = ["LEFT_ARC", "SHIFT", ...]

# later train as model.fit(x_train, y_train)


# later parse as following
features = ['STK_1_POS_xxx', 'STK_3_POS_xxx']
col = []
row = []
data = []
for feature in features:
    if feature in _dictionary:
        col.append(_dictionary[feature])
        row.append(0)
        data.append(1.0)
print "col:", col # col: [1, 3]
print "row:", row # row: [0, 0]
print "data:", data # data: [1.0, 1.0]
np_col = numpy.array(sorted(col))
np_row = numpy.array(row)
np_data = numpy.array(data)
x_test = scipy.sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(_dictionary)))

print x_test
#   (0, 1)	1.0
#   (0, 3)	1.0

# prob_dict = {}
# pred_prob = model.predict_proba(x_test)[0]
