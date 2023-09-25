# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distToGhost = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        distToFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(distToFood) == 0:
            return 100000
        if min(distToGhost) < 2:
            return -100000
        return - sum(distToFood) + min(distToGhost) + successorGameState.getScore() * (10 + 10 / sum(distToFood))


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def maxValue(state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            successors = [state.generateSuccessor(0, action) for action in state.getLegalActions(0)]
            return max([minValue(successor, depth, 1) for successor in successors])

        def minValue(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            successors = [state.generateSuccessor(agentIndex, action) for action in state.getLegalActions(agentIndex)]
            if agentIndex == numAgents - 1:
                return min([maxValue(successor, depth + 1) for successor in successors])
            else:
                return min([minValue(successor, depth, agentIndex + 1) for successor in successors])

        legalActions = gameState.getLegalActions(0)
        scores = [minValue(gameState.generateSuccessor(0, action), 0, 1) for action in legalActions]
        bestScore = max(scores)
        return legalActions[scores.index(bestScore)]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def maxValue(state, depth, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = -float('inf')
            for action in state.getLegalActions(0):
                v = max(v, minValue(state.generateSuccessor(0, action), depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state, depth, agentIndex, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                if agentIndex == numAgents - 1:
                    v = min(v, maxValue(state.generateSuccessor(agentIndex, action), depth + 1, alpha, beta))
                else:
                    v = min(v, minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        legalActions = gameState.getLegalActions(0)
        alpha = -float('inf')
        beta = float('inf')
        selectedAction = None
        bestScore = -float('inf')
        for action in legalActions:
            score = minValue(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                selectedAction = action
            alpha = max(alpha, bestScore)
        return selectedAction




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def avg(list):
            return sum(list) / len(list)

        def maxValue(state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            successors = [state.generateSuccessor(0, action) for action in state.getLegalActions(0)]
            return max([expValue(successor, depth, 1) for successor in successors])

        def expValue(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            successors = [state.generateSuccessor(agentIndex, action) for action in state.getLegalActions(agentIndex)]
            if agentIndex == numAgents - 1:
                return avg([maxValue(successor, depth + 1) for successor in successors])
            else:
                return avg([expValue(successor, depth, agentIndex + 1) for successor in successors])

        legalActions = gameState.getLegalActions(0)
        scores = [expValue(gameState.generateSuccessor(0, action), 0, 1) for action in legalActions]
        bestScore = max(scores)
        return legalActions[scores.index(bestScore)]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFoods = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

    distToGhost = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    distToFood = [manhattanDistance(newPos, food) for food in newFoods]
    distToCapsule = [manhattanDistance(newPos, capsule) for capsule in newCapsules]
    if len(distToFood) == 0:
        return float('inf')
    if min(distToGhost) < 2:
        return float('-inf')
    # if capsule is near, go for it
    if min(distToCapsule) == 0:
        return float('inf')
    if sum(distToFood) < 10:
        return - 10*sum(distToFood) - 55*min(distToFood) + 100 + currentGameState.getScore() * (100 + 10 / sum(distToFood))
    if min(distToFood) > 10:
        return - 10*sum(distToFood) -55*min(distToFood) + 100 + currentGameState.getScore() * (100 + 10 / sum(distToFood))
    if min(distToGhost) > 5 or min(newScaredTimes) > 0:
        return - 10*sum(distToFood) - 15*min(distToFood) - 5*min(distToFood) + 100 + currentGameState.getScore() * (100 + 10 / sum(distToFood)) - 10 * min(distToCapsule)
    else:
        return (- 5*sum(distToFood) + 10 * min(distToGhost) - 30*min(distToFood)
                + currentGameState.getScore() * (100 + 10 / sum(distToFood)) - 10 * min(distToCapsule))



# Abbreviation
better = betterEvaluationFunction
