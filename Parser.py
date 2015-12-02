import sys
import time
import Constants as C
from Oracle import Oracle
from Perceptron import PerceptronModel
# from DeepLearningModel import DeepLearningModel
from SvmModel import SvmModel
from Transition import Transition
from collections import defaultdict


class Parser:
    def __init__(self, labeled):
        self.labeled = labeled
        self.posTypes = []
        self.labelTypes = []
        # self.model = PerceptronModel(self.labeled)
        # self.model = DeepLearningModel(self.labeled, self.posTypes, self.labelTypes)
        self.model = SvmModel(self.labeled)

    def initialize(self, sentence):
        #            ID    FORM   LEMMA  CPOSTAG  POSTAG  FEATS   HEAD  DEPREL  PHEAD   PDEPREL
        self.root = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '-1', 'ROOT', 'ROOT', 'ROOT']
        self.buff = [self.root] + list(reversed(sentence))  # buff = [ROOT, N, N-1, ... 1]
        self.stack = list()
        self.arcs = {}  # {dependentId: headId}
        self.labels = {}  # {dependentId: dependentLabel}
        self.wordsThatHaveHeads = {}  # {wordId: word}
        self.transitions = list()
        self.dependentIDs = defaultdict(list)  # {headId: [dependentId, ...]}

        # Calculate the leftmost and rightmost children for each node in the sentence
        # Note: At test time this data is not used.
        for word in sentence:
            self.dependentIDs[word[C.HEAD]].append(word[C.ID])

    # Record types that were seen
    def trackTypes(self, sentence):
        for word in sentence:
            if word[C.POSTAG] not in self.posTypes:
                self.posTypes.append(word[C.POSTAG])
            if word[C.DEPREL] not in self.labelTypes:
                self.labelTypes.append(word[C.DEPREL])

    # This function should take a transition object and apply to the current parser state. It need not return anything.
    def execute_transition(self, transition):
        if transition.transitionType == Transition.Shift:
            self.stack.append(self.buff.pop())
        else:
            head = self.stack[-1] if transition.transitionType == Transition.LeftArc else self.stack[-2]
            dependent = self.stack[-2] if transition.transitionType == Transition.LeftArc else self.stack[-1]
            self.arcs[dependent[C.ID]] = head[C.ID]
            self.labels[dependent[C.ID]] = transition.label
            self.stack.remove(dependent)
            self.wordsThatHaveHeads[dependent[C.ID]] = dependent
        self.transitions.append(transition)

    @staticmethod
    def load_corpus(filename):
        print >> sys.stderr, 'Loading treebank from %s' % filename
        corpus = []
        sentence = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    corpus.append(sentence)
                    sentence = []
                else:
                    word = line.split('\t')
                    sentence.append(word)
        print >> sys.stderr, 'Loaded %d sentences' % len(corpus)
        return corpus

    def output(self, sentence):
        for word in sentence:
            word[C.HEAD] = self.arcs.get(word[C.ID], '0')
            label = self.labels.get(word[C.ID], '_')
            word[C.DEPREL] = label if label is not None else '_'
            print '\t'.join(word)
        print

    def train(self, trainingSet):
        corpus = Parser.load_corpus(trainingSet)
        oracle = Oracle()
        for sentence in corpus:
            self.initialize(sentence)
            self.trackTypes(sentence)
            while len(self.buff) > 0 or len(self.stack) > 1:
                transition = oracle.getTransition(self.stack, self.buff, self.dependentIDs, self.arcs, self.labeled)
                self.model.learn(transition, self.stack, self.buff, self.labels, self.transitions, self.arcs, self.wordsThatHaveHeads)
                self.execute_transition(transition)

    def parse(self, testSet):
        corpus = Parser.load_corpus(testSet)
        for sentence in corpus:
            self.initialize(sentence)
            while len(self.buff) > 0 or len(self.stack) > 1:
                _, transition = self.model.predict(self.stack, self.buff, self.labels, self.transitions, self.arcs, self.wordsThatHaveHeads)
                self.execute_transition(transition)
            self.output(sentence)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled', '-l', action='store_true')
    parser.add_argument('trainingSet', help='Training treebank')
    parser.add_argument('testSet', help='Dev/test treebank')
    args = parser.parse_args()

    startTime = time.time()

    p = Parser(args.labeled)
    p.train(args.trainingSet)
    p.parse(args.testSet)

    print >> sys.stderr,  "Elapsed Time: %s seconds" % (time.time() - startTime)
