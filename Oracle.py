import sys
from Transition import Transition


class Oracle:

    # This function should return a Transition object representing the correct action to take according to the oracle.
    def getTransition(self, stack, buff, leftmostChildren, rightmostChildren, arcs, labeled):
        if len(stack) > 1:
            topOfStack = stack[-1]
            belowTopOfStack = stack[-2]

            # if stack[-1] is head of stack[-2] and all children of stack[-2] are attached to it and stack[-2] is not the root
            if topOfStack[0] == belowTopOfStack[6]\
                    and self.hasAllChildrenAttached(belowTopOfStack[0], leftmostChildren, rightmostChildren, arcs)\
                    and belowTopOfStack[1] != "ROOT":
                return Transition(Transition.LeftArc, belowTopOfStack[7] if labeled else None)

            # if stack[-2] is head of stack[-1] and all children of stack[-1] are attached to it and stack[-1] is not the root
            if belowTopOfStack[0] == topOfStack[6]\
                    and self.hasAllChildrenAttached(topOfStack[0], leftmostChildren, rightmostChildren, arcs)\
                    and topOfStack[1] != "ROOT":
                return Transition(Transition.RightArc, topOfStack[7] if labeled else None)

        return Transition(Transition.Shift, None)

    def hasAllChildrenAttached(self, wordIndex, leftmostChildren, rightmostChildren, arcs):
        children = leftmostChildren.get(wordIndex, []) + rightmostChildren.get(wordIndex, [])
        for child in children:
            if child not in arcs:
                return False

        return True
