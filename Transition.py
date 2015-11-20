#!/usr/bin/env python


class Transition:
    # Transition types
    Shift = 0
    LeftArc = 1
    RightArc = 2

    def __init__(self, transitionType, label):
        self.transitionType = transitionType
        self.label = label

    def __str__(self):
        return 'Transition of type %d with label %s' % (self.transitionType, self.label)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.transitionType == other.transitionType and self.label == other.label

    def __ne__(self, other):
        return not (self == other)
