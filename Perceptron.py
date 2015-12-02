import sys
import math
import Constants as C
from Transition import Transition
from collections import defaultdict


def dot(features, weights):
    score = 0.0
    for key in set(features) & set(weights):
        score += features[key] * weights[key]
    return score


class PerceptronModel:
    def __init__(self, labeled):
        self.labeled = labeled
        self.learning_rate = 1.0
        self.weights = defaultdict(float)
        self.label_set = ('abbrev acomp advcl advmod amod appos ' + \
                          'attr aux auxpass cc ccomp complm conj cop csubj ' + \
                          'dep det dobj expl infmod iobj mark measure neg ' + \
                          'nn nsubj nsubjpass null num number parataxis partmod ' + \
                          'pcomp pobj poss possessive preconj pred predet ' + \
                          'prep prt punct purpcl quantmod rcmod rel tmod ' + \
                          'xcomp').split() if labeled else [None]

    def extract_features(self, transition, stack, buff, labels, previous_transitions, arcs, processedWords):
        features = defaultdict(float)

        tType = transition.transitionType
        tLabel = transition.label

        # Top three POS tags and forms from the stack
        for i in range(3):
            if i >= len(stack):
                break
            s = stack[-(i + 1)]
            pos = s[C.POSTAG]
            features['transition=%d,s%d.pos=%s' % (tType, i, pos)] = 1
            # (1)
            form = s[C.FORM]
            features['transition=%d,s%d.form=%s' % (tType, i, form.lower())] = 1
            features['transition=%d,s%d.pos=%s,s%d.form=%s' % (tType, i, pos, i, form.lower())] = 1

        # Top three POS tags and forms from the buffer
        for i in range(3):
            if i >= len(buff):
                break
            b = buff[-(i + 1)]
            pos = b[C.POSTAG]
            features['transition=%d,b%d.pos=%s' % (tType, i, pos)] = 1
            # (1)
            form = b[C.FORM]
            features['transition=%d,b%d.form=%s' % (tType, i, form.lower())] = 1
            features['transition=%d,b%d.pos=%s,b%d.form=%s' % (tType, i, pos, i, form.lower())] = 1

        # Bigrams
        if len(buff) > 0 and len(stack) > 0:
            # POS from stack[-1] and buff[-1]
            s0 = stack[-1]
            s0pos = s0[C.POSTAG]
            s0form = s0[C.FORM].lower()
            b0 = buff[-1]
            b0pos = b0[C.POSTAG]
            b0form = b0[C.FORM].lower()
            features['transition=%d,s0.pos=%s,b0.pos=%s' % (tType, s0pos, b0pos)] = 1
            # (1)
            features['transition=%d,s0.form=%s,b0.form=%s' % (tType, s0form, b0form)] = 1
            features['transition=%d,b0.pos=%s,b0.form=%s,s0.pos=%s' % (tType, b0pos, b0form, s0pos)] = 1
            features['transition=%d,b0.pos=%s,b0.form=%s,s0.form=%s' % (tType, b0pos, b0form, s0form)] = 1
            features['transition=%d,s0.pos=%s,s0.form=%s,b0.pos=%s' % (tType, s0pos, s0form, b0pos)] = 1
            features['transition=%d,s0.pos=%s,s0.form=%s,b0.form=%s' % (tType, s0pos, s0form, b0form)] = 1
            features['transition=%d,s0.pos=%s,s0.form=%s,b0.pos=%s,b0.form=%s' % (tType, s0pos, s0form, b0pos, b0form)] = 1

            if len(stack) > 1:
                # POS from stack[-1], stack[-2], and buff[-1]
                features['transition=%d,s0.pos=%s,s1.pos=%s,b0.pos=%s' % (tType, stack[-1][C.POSTAG], stack[-2][C.POSTAG], buff[-1][C.POSTAG])] = 1
            if len(buff) > 1:
                # POS from stack[-1], buff[-1], and buff[-2]
                features['transition=%d,s0.pos=%s,b0.pos=%s,b1.pos=%s' % (tType, stack[-1][C.POSTAG], buff[-1][C.POSTAG], buff[-2][C.POSTAG])] = 1

        # if len(buff) > 2:
        #     # POS from buff[-1], buff[-2], and buff[-3]
        #     features['transition=%d,b0.pos=%s,b1.pos=%s,b2.pos=%s' % (tType, buff[-1][C.POSTAG], buff[-2][C.POSTAG], buff[-3][C.POSTAG])] = 1
        #
        # if len(buff) > 3:
        #     # POS from buff[-2], buff[-3], and buff[-4]
        #     features['transition=%d,b1.pos=%s,b2.pos=%s,b3.pos=%s' % (tType, buff[-2][C.POSTAG], buff[-3][C.POSTAG], buff[-4][C.POSTAG])] = 1

        # Previous transition type
        if len(previous_transitions) > 0:
            features['transition=%d,prev_transition0=%d' % (tType, previous_transitions[-1].transitionType)] = 1
            # if len(previous_transitions) > 1:
            #     features['transition=%d,prev_transition0=%d,prev_transition1=%d' % (tType, previous_transitions[-1].transitionType, previous_transitions[-2].transitionType)] = 1
        else:
            features['transition=%d,prev_transition0=None' % (tType)] = 1

        # Histories (2)
        if len(stack) > 0:
            s0 = stack[-1]
            s0pos = s0[C.POSTAG]
            s0form = s0[C.FORM].lower()
            leftDependentIds, rightDependentIds = self.getDependentsIds(s0[C.ID], arcs)
            for i in range(2):
                if i >= len(leftDependentIds):
                    break
                leftDependent = processedWords[leftDependentIds[-(i + 1)]]
                features['transition=%d,s0.pos=%s,s0l%d.pos=%s' % (tType, s0pos, i, leftDependent[C.POSTAG])] = 1
                features['transition=%d,s0.form=%s,s0l%d.form=%s' % (tType, s0form, i, leftDependent[C.FORM].lower())] = 1
            if len(leftDependentIds) > 1:
                features['transition=%d,s0.pos=%s,s0l0.pos=%s,s0l1.pos=%s' % (tType, s0pos, processedWords[leftDependentIds[-1]][C.POSTAG], processedWords[leftDependentIds[-2]][C.POSTAG])] = 1
                features['transition=%d,s0.form=%s,s0l0.form=%s,s0l1.form=%s' % (tType, s0form, processedWords[leftDependentIds[-1]][C.FORM].lower(), processedWords[leftDependentIds[-2]][C.FORM].lower())] = 1
            for i in range(2):
                if i >= len(rightDependentIds):
                    break
                rightDependent = processedWords[rightDependentIds[-(i + 1)]]
                features['transition=%d,s0r%d.pos=%s' % (tType, i, rightDependent[C.POSTAG])] = 1
                features['transition=%d,s0r%d.form=%s' % (tType, i, rightDependent[C.FORM].lower())] = 1
            if len(rightDependentIds) > 1:
                features['transition=%d,s0.pos=%s,s0r0.pos=%s,s0r1.pos=%s' % (tType, s0pos, processedWords[rightDependentIds[-1]][C.POSTAG], processedWords[rightDependentIds[-2]][C.POSTAG])] = 1
                features['transition=%d,s0.form=%s,s0r0.form=%s,s0r1.form=%s' % (tType, s0form, processedWords[rightDependentIds[-1]][C.FORM].lower(), processedWords[rightDependentIds[-2]][C.FORM].lower())] = 1

        # Bias feature
        features['transition=%d' % (tType)] = 1

        if self.labeled:
            # Action and label pair
            features['transition=%d,label=%s' % (tType, tLabel)] = 1

            if len(stack) > 0:
                # label of stack[-1]
                features['transition=%d,s0.label=%s' % (tType, stack[-1][C.DEPREL])] = 1
                # label and pos of stack[-1]
                features['transition=%d,s0.pos=%s,s0.label=%s' % (tType, stack[-1][C.POSTAG], stack[-1][C.DEPREL])] = 1

            # Label bias
            features['label=%s' % (tLabel)] = 1

        return features

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

    def possible_transitions(self, stack, buff):
        possible_transitions = []
        if len(buff) >= 1:
            possible_transitions.append(Transition(Transition.Shift, None))
        if len(stack) >= 2:
            for label in self.label_set:
                possible_transitions.append(Transition(Transition.LeftArc, label))
                possible_transitions.append(Transition(Transition.RightArc, label))
        assert len(possible_transitions) > 0
        return possible_transitions

    def update(self, correct_features, predicted_features):
        keys = set(correct_features) | set(predicted_features)
        for key in keys:
            c = correct_features.get(key, 0.0)
            p = predicted_features.get(key, 0.0)
            self.weights[key] += (c - p) * self.learning_rate
            if self.weights[key] == 0.0:
                del self.weights[key]

    def learn(self, correct_transition, stack, buff, labels, previous_transitions, arcs, processedWords):
        correct_features = None
        best_features = None
        best_score = None
        best_transition = None
        for transition in self.possible_transitions(stack, buff):
            features = self.extract_features(transition, stack, buff, labels, previous_transitions, arcs, processedWords)
            score = dot(features, self.weights)
            if best_score is None or score > best_score:
                best_score = score
                best_transition = transition
                best_features = features
            if transition == correct_transition:
                correct_features = features

        if best_transition != correct_transition:
            assert best_features is not None
            assert correct_features is not None
            self.update(correct_features, best_features)

    def predict(self, stack, buff, labels, previous_transitions, arcs, processedWords):
        best_score = None
        best_transition = None
        for transition in self.possible_transitions(stack, buff):
            features = self.extract_features(transition, stack, buff, labels, previous_transitions, arcs, processedWords)
            score = dot(features, self.weights) # TODO: use average weights?
            if best_score is None or score > best_score:
                best_score = score
                best_transition = transition
        return (best_score, best_transition)
