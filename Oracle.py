import sys
import Constants as C
from Transition import Transition


class Oracle:

    # This function should return a Transition object representing the correct action to take according to the oracle.
    def getTransition(self, stack, buff, dependentIDs, arcs, labeled):
        if len(stack) > 1:
            firstWord = stack[-1]
            secondWord = stack[-2]

            # if stack[-1] is head of stack[-2] and stack[-2] has all dependents attached and stack[-2] is not the root
            if firstWord[C.ID] == secondWord[C.HEAD]\
                    and self.hasAllDependentsAttached(secondWord[C.ID], dependentIDs, arcs)\
                    and secondWord[C.FORM] != "ROOT":
                return Transition(Transition.LeftArc, secondWord[C.DEPREL] if labeled else None)

            # if stack[-2] is head of stack[-1] and stack[-1] has all dependents attached and stack[-1] is not the root
            if secondWord[C.ID] == firstWord[C.HEAD]\
                    and self.hasAllDependentsAttached(firstWord[C.ID], dependentIDs, arcs)\
                    and firstWord[C.FORM] != "ROOT":
                return Transition(Transition.RightArc, firstWord[C.DEPREL] if labeled else None)

        return Transition(Transition.Shift, None)

    @staticmethod
    def hasAllDependentsAttached(wordId, dependentIDs, arcs):
        for dependentId in dependentIDs[wordId]:
            if dependentId not in arcs:
                return False  # not all arcs that connect to wordId as head have been created yet

        return True
