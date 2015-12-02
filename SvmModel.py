import sys
import operator
import Constants as C
import numpy
import scipy
from sklearn import svm
from Transition import Transition


class SvmModel:
    def __init__(self, labeled):
        self.labeled = labeled
        self.featureDict = {}  # dict of every feature seen {"s0.pos=xxx": position index in normalized feature vect}
        self.features = []  # list of featureVector ["s0.pos=xxx", ...] for every transitionType
        self.labels = []  # list of transitionType (0 or 1 or 2) for each featureVector
        self.model = svm.SVC(
            kernel='poly',
            degree=2,
            coef0=0,
            gamma=0.2,
            C=0.5,
            # verbose=True,
            probability=True)
        self.isTrained = False  # boolean for keeping track of whether model has been trained or not

    def createFeatureVector(self, stack, buff, labels, previous_transitions, arcs, wordsThatHaveHeads):
        featureVector = []

        # Top two POS tags from the stack
        for i in range(4):
            if i < len(stack):
                s = stack[-(i + 1)]
                pos = s[C.POSTAG]
                form = s[C.FORM].lower()
                featureVector.append('s[%d].pos=%s' % (i, pos))
                # featureVector.append('s[%d].form=%s' % (i, form))
                # featureVector.append('s[%d].pos=%s,s[%d].form=%s' % (i, pos, i, form))

        # Next four POS tags from the buffer
        for i in range(4):
            if i < len(buff):
                b = buff[-(i + 1)]
                pos = b[C.POSTAG]
                form = b[C.FORM].lower()
                # featureVector.append('b[%d].pos=%s' % (i, pos))
                # featureVector.append('b[%d].form=%s' % (i, form))
                # featureVector.append('b[%d].pos=%s,b[%d].form=%s' % (i, pos, i, form))

        for i in range(2):
            if i < len(stack) and i < len(buff):
                s = stack[-(i + 1)]
                sPos = s[C.POSTAG]
                b = buff[-(i + 1)]
                bPos = b[C.POSTAG]
                # featureVector.append('s[%d].pos=%s,b[%d].pos=%s' % (i, sPos, i, bPos))

        # Histories
        for i in range(1):
            # Stack histories
            if i < len(stack):
                s = stack[-(i + 1)]
                sPos = s[C.POSTAG]
                sForm = s[C.FORM].lower()
                leftDependentIds, rightDependentIds = self.getDependentsIds(s[C.ID], arcs)

                # Nearest left dependents of stack
                for j in range(2):
                    if j < len(leftDependentIds):
                        l = wordsThatHaveHeads[leftDependentIds[-(j + 1)]]
                        lPos = l[C.POSTAG]
                        lForm = l[C.FORM].lower()
                        featureVector.append('s[%d]l[%d].pos=%s' % (i, j, lPos))
                        featureVector.append('s[%d]l[%d].form=%s' % (i, j, lForm))
                        # featureVector.append('s[%d].pos=%s,s[%d]l[%d].pos=%s' % (i, sPos, i, j, lPos))
                        # featureVector.append('s[%d].pos=%s,s[%d]l[%d].form=%s' % (i, sPos, i, j, lForm))
                if len(leftDependentIds) > 1:
                    l0 = wordsThatHaveHeads[leftDependentIds[-1]]
                    l0Pos = l0[C.POSTAG]
                    l0Form = l0[C.FORM].lower()
                    l1 = wordsThatHaveHeads[leftDependentIds[-2]]
                    l1Pos = l1[C.POSTAG]
                    l1Form = l1[C.FORM].lower()
                    # featureVector.append('s[%d].pos=%s,s[%d]l[0].pos=%s,s[%d]l[1].pos=%s' % (i, sPos, i, l0Pos, i, l1Pos))
                    # featureVector.append('s[%d].form=%s,s[%d]l[0].form=%s,s[%d]l[1].form=%s' % (i, sForm, i, l0Form, i, l1Form))

                # Nearest right dependents of stack
                for j in range(2):
                    if j < len(rightDependentIds):
                        r = wordsThatHaveHeads[rightDependentIds[j]]
                        rPos = r[C.POSTAG]
                        rForm = r[C.FORM].lower()
                        featureVector.append('s[%d]r[%d].pos=%s' % (i, j, rPos))
                        featureVector.append('s[%d]r[%d].form=%s' % (i, j, rForm))
                        # featureVector.append('s[%d].pos=%s,s[%d]r[%d].pos=%s' % (i, sPos, i, j, rPos))
                        # featureVector.append('s[%d].pos=%s,s[%d]r[%d].form=%s' % (i, sPos, i, j, rForm))
                if len(rightDependentIds) > 1:
                    r0 = wordsThatHaveHeads[rightDependentIds[0]]
                    r0Pos = r0[C.POSTAG]
                    r0Form = r0[C.FORM].lower()
                    r1 = wordsThatHaveHeads[rightDependentIds[1]]
                    r1Pos = r1[C.POSTAG]
                    r1Form = r1[C.FORM].lower()
                    # featureVector.append('s[%d].pos=%s,s[%d]r[0].pos=%s,s[%d]r[1].pos=%s' % (i, sPos, i, r0Pos, i, r1Pos))
                    # featureVector.append('s[%d].form=%s,s[%d]r[0].form=%s,s[%d]r[1].form=%s' % (i, sForm, i, r0Form, i, r1Form))

            # Buffer histories  # TODO: probably won't do anything since stuff in buff isn't attached to anything!
            if i < len(buff):
                b = buff[-(i + 1)]
                bPos = b[C.POSTAG]
                bForm = b[C.FORM].lower()
                leftDependentIds, rightDependentIds = self.getDependentsIds(b[C.ID], arcs)

                # Nearest left dependents of buff
                for j in range(2):
                    if j < len(leftDependentIds):
                        l = wordsThatHaveHeads[leftDependentIds[-(j + 1)]]
                        lPos = l[C.POSTAG]
                        lForm = l[C.FORM].lower()
                        featureVector.append('b[%d]l[%d].pos=%s' % (i, j, lPos))
                        featureVector.append('b[%d]l[%d].form=%s' % (i, j, lForm))
                        # featureVector.append('s[%d].pos=%s,s[%d]l[%d].pos=%s' % (i, sPos, i, j, lPos))
                        # featureVector.append('s[%d].pos=%s,s[%d]l[%d].form=%s' % (i, sPos, i, j, lForm))
                if len(leftDependentIds) > 1:
                    l0 = wordsThatHaveHeads[leftDependentIds[-1]]
                    l0Pos = l0[C.POSTAG]
                    l0Form = l0[C.FORM].lower()
                    l1 = wordsThatHaveHeads[leftDependentIds[-2]]
                    l1Pos = l1[C.POSTAG]
                    l1Form = l1[C.FORM].lower()
                    featureVector.append('b[%d].pos=%s,b[%d]l[0].pos=%s,b[%d]l[1].pos=%s' % (i, bPos, i, l0Pos, i, l1Pos))
                    featureVector.append('b[%d].form=%s,b[%d]l[0].form=%s,b[%d]l[1].form=%s' % (i, bForm, i, l0Form, i, l1Form))

        # Previous transition type
        if len(previous_transitions) > 0:
            featureVector.append('t[0].type=%d' % (previous_transitions[-1].transitionType))
        else:
            featureVector.append('t[0].type=%d' % (-1))

        return featureVector

    # Convert and concatenate all original featureVectors into one big m*n sparse matrix
    def getNormalizedFeatures(self, features):
        normalizedFeatures = []
        for feature in features:
            normalizedFeatures.append(self.getNormalizedFeatureVector(feature))

        return scipy.sparse.vstack(normalizedFeatures)

    # Convert a featureVector into a 1*n sparse matrix
    def getNormalizedFeatureVector(self, featureVector):
        col = []
        row = []
        data = []
        for feature in featureVector:
            if feature in self.featureDict:
                col.append(self.featureDict[feature])
                row.append(0)
                data.append(1.0)
        np_col = numpy.array(sorted(col))
        np_row = numpy.array(row)
        np_data = numpy.array(data)
        return scipy.sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(self.featureDict)))

    # Accumulate feature vectors and labels
    def learn(self, correct_transition, stack, buff, labels, previous_transitions, arcs, wordsThatHaveHeads):
        featureVector = self.createFeatureVector(stack, buff, labels, previous_transitions, arcs, wordsThatHaveHeads)
        # Keep track of every newly seen feature and assign number to it
        for feature in featureVector:
            self.featureDict.setdefault(feature, len(self.featureDict))

        self.features.append(featureVector)
        self.labels.append(correct_transition.transitionType)

    # Predict a transition based on the given configuration state
    def predict(self, stack, buff, labels, previous_transitions, arcs, wordsThatHaveHeads):
        # First train the model if it hasn't been done so already
        if not self.isTrained:
            print >> sys.stderr, "Start training (#rows="+str(len(self.labels))+", #features="+str(len(self.featureDict))+")"
            self.model.fit(self.getNormalizedFeatures(self.features), self.labels)
            print >> sys.stderr, "Finished training"
            self.isTrained = True

        # Get order of best possible transitions according to the number of votes each label got
        featureVector = self.createFeatureVector(stack, buff, labels, previous_transitions, arcs, wordsThatHaveHeads)
        decisionFunction = self.model.decision_function(self.getNormalizedFeatureVector(featureVector))[0]
        votesPerLabel = {}
        k = 0
        for i in range(len(self.model.classes_)):
            for j in range(i + 1, len(self.model.classes_)):
                if decisionFunction[k] > 0:
                    votesPerLabel.setdefault(i, 0)
                    votesPerLabel[i] += 1
                else:
                    votesPerLabel.setdefault(j, 0)
                    votesPerLabel[j] += 1
            k += 1
        sortedVotesPerLabel = sorted(votesPerLabel.items(), key=operator.itemgetter(1), reverse=True)

        # Get the best transition
        bestTransitionType = None
        for (labelIndex, numVotes) in sortedVotesPerLabel:
            predictedTransitionType = self.model.classes_[labelIndex]
            if predictedTransitionType == Transition.LeftArc and self.canPerformLeftArc(stack, buff):
                bestTransitionType = Transition.LeftArc
                break
            elif predictedTransitionType == Transition.RightArc and self.canPerformRightArc(stack, buff):
                bestTransitionType = Transition.RightArc
                break
            elif predictedTransitionType == Transition.Shift and self.canPerformShift(buff):
                bestTransitionType = Transition.Shift
                break

        if bestTransitionType is None:
            if self.canPerformShift(buff):  # If type Transition.Shift had zero votes but can still be performed
                bestTransitionType = Transition.Shift
            else:
                print >> sys.stderr, 'No transition type available.'
                raise RuntimeError('No transition type available.')

        return (0, Transition(bestTransitionType, None))

    def canPerformShift(self, buff):
        if len(buff) <= 0:
            return False

        return True

    def canPerformLeftArc(self, stack, buff):
        if len(stack) <= 1:
            return False

        return True

    def canPerformRightArc(self, stack, buff):
        if len(stack) <= 1:
            return False

        if stack[-1][C.FORM] == "ROOT":
            return False

        return True

    def getDependentsIds(self, id, arcs):
        leftDependents = []
        rightDependents = []

        for dependentId, headId in arcs.items():
            if id == headId:
                if int(dependentId) < int(id):
                    leftDependents.append(dependentId)
                else:
                    rightDependents.append(dependentId)

        return sorted(leftDependents), sorted(rightDependents)


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




# def runSvmExample():
#     _dictionary = {}  # {'STK_4_POS_xxx': 4, 'STK_5_POS_xxx': 5, 'STK_1_POS_xxx': 1, 'STK_2_POS_xxx': 2, 'STK_3_POS_xxx': 3, 'STK_0_POS_xxx': 0}
#
#     def getNormalizedFeatures(features):
#         normalizedFeatures = []
#         for feature in features:
#             normalizedFeatures.append(getNormalizedFeatureVector(feature))
#
#         return normalizedFeatures
#
#     def getNormalizedFeatureVector(featureVector):
#         col = []
#         row = []
#         data = []
#         for feature in featureVector:
#             if feature in _dictionary:
#                 col.append(_dictionary[feature])
#                 row.append(0)
#                 data.append(1.0)
#         np_col = numpy.array(sorted(col))
#         np_row = numpy.array(row)
#         np_data = numpy.array(data)
#         return scipy.sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(_dictionary)))
#
#     def _convert_to_binary_features(features):
#         """
#         :param features: list of feature string which is needed to convert to binary features
#         :type features: list(str)
#         :return : string of binary features in libsvm format  which is 'featureID:value' pairs
#         """
#         unsorted_result = []
#         for feature in features:
#             _dictionary.setdefault(feature, len(_dictionary))
#             unsorted_result.append(_dictionary[feature])
#
#         # Default value of each feature is 1.0
#         return ' '.join(str(featureID) + ':1.0' for featureID in sorted(unsorted_result))
#
#     features1 = ['STK_0_POS_xxx', 'STK_1_POS_xxx', 'STK_2_POS_xxx', 'STK_3_POS_xxx']  # extract_features
#     binary_features1 = _convert_to_binary_features(features1)  # "0:1.0 1:1.0 2:1.0 3:1.0"   =>   binary_features
#     features2 = ['STK_2_POS_xxx', 'STK_3_POS_xxx', 'STK_4_POS_xxx', 'STK_5_POS_xxx']
#     binary_features2 = _convert_to_binary_features(features2)  # 2:1.0 3:1.0 4:1.0 5:1.0
#     features3 = ['STK_6_POS_xxx', 'STK_7_POS_xxx']
#     binary_features3 = _convert_to_binary_features(features3)
#     labels = [0, 1, 2]
#
#     # gives out X, Y as
#     # (X, Y) = (binary_features, transition)
#     # training_seq = ["LEFT_ARC", "SHIFT", ...]
#
#     # later train as model.fit(x_train, y_train)
#     model = svm.SVC(
#         kernel='poly',
#         degree=2,
#         coef0=0,
#         gamma=0.2,
#         C=0.5,
#         # verbose=True,
#         probability=True)
#
#     # print getNormalizedFeatureVector(features1)
#     concatenated = scipy.sparse.vstack([getNormalizedFeatureVector(features1), getNormalizedFeatureVector(features2),
#                                         getNormalizedFeatureVector(features3)])
#     print concatenated
#     model.fit(concatenated, labels)
#
#     # later parse as following
#     features = ['STK_1_POS_xxx', 'STK_3_POS_xxx']
#     col = []
#     row = []
#     data = []
#     for feature in features:
#         if feature in _dictionary:
#             col.append(_dictionary[feature])
#             row.append(0)
#             data.append(1.0)
#     print "col:", col  # col: [1, 3]
#     print "row:", row  # row: [0, 0]
#     print "data:", data  # data: [1.0, 1.0]
#     np_col = numpy.array(sorted(col))
#     np_row = numpy.array(row)
#     np_data = numpy.array(data)
#     x_test = scipy.sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(_dictionary)))
#
#     print x_test
#     #   (0, 1)	1.0
#     #   (0, 3)	1.0
#
#     x_test = getNormalizedFeatureVector(features1)
#     print "predicted:", model.predict(x_test)
#
#     prob_dict = {}
#     pred_prob = model.predict_proba(x_test)[0]
#     print pred_prob
#     for i in range(len(pred_prob)):
#         prob_dict[i] = pred_prob[i]
#     sorted_prob = sorted(prob_dict.items(), key=operator.itemgetter(1), reverse=True)
#     print "sorted_prob:", sorted_prob
#
#     dec_func = model.decision_function(x_test)[0]
#     print "dec_func:", dec_func
#     votes = {}
#     k = 0
#     for i in range(len(model.classes_)):
#         for j in range(i + 1, len(model.classes_)):
#             if dec_func[k] > 0:
#                 votes.setdefault(i, 0)
#                 votes[i] += 1
#             else:
#                 votes.setdefault(j, 0)
#                 votes[j] += 1
#         k += 1
#     # Sort votes according to the values
#     sorted_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
#     print "sorted_votes:", sorted_votes
#
#
#     # clf = svm.LinearSVC()
#     # clf.fit(concatenated, labels)
#     # print clf.decision_function(x_test)
#
#
# runSvmExample()
