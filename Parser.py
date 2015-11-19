import sys
import time
from Oracle import Oracle
from Perceptron import PerceptronModel
from Transition import Transition


class Parser:
    def __init__(self, labeled):
        self.labeled = labeled

    def initialize(self, sentence):
        self.root = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '-1', 'ROOT', 'ROOT', 'ROOT']
        self.buff = [self.root] + list(reversed(sentence))
        self.stack = list()
        self.arcs = {}
        self.labels = {}
        self.transitions = list()
        self.leftmostChildren = {}
        self.rightmostChildren = {}
        # TODO: Calculate the leftmost and rightmost children for each node in the sentence
        # Note: At test time this data is not used.
        assert False, 'Please finish implementing this function!'

    def execute_transition(self, transition):
        # This function should take a transition object and apply to the current parser state. It need not return anything.
        assert False, 'Please implement this function!'

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
        for token in sentence:
            head = self.arcs.get(token[0], '0')
            label = self.labels.get(token[0], '_')
            label = label if label is not None else '_'
            token[6] = head
            token[7] = label
            print '\t'.join(token)
        print

    def train(self, trainingSet, model):
        corpus = Parser.load_corpus(trainingSet)
        oracle = Oracle()
        for sentence in corpus:
            self.initialize(sentence)
            while len(self.buff) > 0 or len(self.stack) > 1:
                transition = oracle.getTransition(self.stack, self.buff, \
                                                  self.leftmostChildren, self.rightmostChildren, \
                                                  self.arcs, self.labeled)
                model.learn(transition, self.stack, self.buff, \
                            self.labels, self.transitions)
                self.execute_transition(transition)

    def parse(self, testSet, model):
        corpus = Parser.load_corpus(testSet)
        for sentence in corpus:
            self.initialize(sentence)
            while len(self.buff) > 0 or len(self.stack) > 1:
                _, transition = model.predict(self.stack, self.buff, \
                                              self.labels, self.transitions)
                self.execute_transition(transition)
            self.output(sentence)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled', '-l', action='store_true')
    parser.add_argument('trainingset', help='Training treebank')
    parser.add_argument('testset', help='Dev/test treebank')
    args = parser.parse_args()

    p = Parser(args.labeled)
    model = PerceptronModel(args.labeled)

    p.train(args.trainingset, model)
    p.parse(args.testset, model)
