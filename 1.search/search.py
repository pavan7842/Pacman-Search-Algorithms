# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import aStarPriorityQueue, PriorityQueueWithFunction, manhattanDistance
from datetime import datetime
from game import Directions,Actions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print(problem)

    openSet = util.Stack()#openSet is a Stack in this case, to maintain fringe, unvisited nodes
    closedSet = []#to maintain visited nodes
    actions = []#maintain a list of actions, so you dont have to backtrack to reach the goal for pacman
    startState = problem.getStartState()

    if problem.isGoalState(startState):#return empty list, no actions after this goal reached
        return []

    openSet.push((startState,actions))#push start state and actions to reach state to stack

    while not openSet.isEmpty():
        current_state,actions = openSet.pop()
        if current_state not in closedSet:
            closedSet.append(current_state)
            if problem.isGoalState(current_state):#return actions, reached goal
                return actions
            for nextState, action,cost in problem.getSuccessors(current_state):#fetch new actions and add to actions
                actionsToState = list(actions)
                actionsToState.append(action)
                openSet.push((nextState,actionsToState))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    openSet = util.Queue()#openSet is a Queue in this case, to maintain fringe, unvisited nodes
    ##rest of the code same as dfs
    closedSet = []#to maintain visited nodes
    actions = []#maintain a list of actions, so you dont have to backtrack to reach the goal for pacman
    startState = problem.getStartState()

    if problem.isGoalState(startState):#return empty list, no actions after this goal reached
        return []

    openSet.push((startState,actions))#push start state and actions to reach state to stack

    while not openSet.isEmpty():
        current_state,actions = openSet.pop()
        if current_state not in closedSet:
            closedSet.append(current_state)
            if problem.isGoalState(current_state):#return actions, reached goal
                return actions
            for nextState, action,cost in problem.getSuccessors(current_state):#fetch new actions and add to actions
                actionsToState = list(actions)
                actionsToState.append(action)
                openSet.push((nextState,actionsToState))
    return actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    openSet = util.PriorityQueue()
    closedSet = []
    actions = []
    cost=0
    startState = problem.getStartState()

    if problem.isGoalState(startState):
        return []

    openSet.push((startState, actions, cost), cost)#saving priority in the node as pop doesn't return priority 

    while not openSet.isEmpty():
        currentState, actions, parentNodeCost = openSet.pop()
        if currentState not in closedSet:
            closedSet.append(currentState)
            if problem.isGoalState(currentState):
                return actions
            for nextState, action, cost in problem.getSuccessors(currentState):
                updatedPriority = cost + parentNodeCost
                actionsToState = list(actions)
                actionsToState.append(action)
                openSet.push((nextState, actionsToState, updatedPriority), updatedPriority)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def getActions(path):
    actions = []
    x,y = path[0]
    for i in range(1,len(path)):
        xi,yi = path[i]
        #X val compare
        if x > xi:
            actions.append(Directions.WEST)
        elif x<xi:
            actions.append(Directions.EAST)
        #y val compare
        if y<yi:
            actions.append(Directions.NORTH)
        elif y>yi:
            actions.append(Directions.SOUTH)
        x = xi
        y = yi
    return actions

def getDirections(way):
    direct = []
    x,y = way[0]
    way_range = range(1,len(way))
    for i in way_range:
        xi,yi = way[i]
        #X val compare
        if x > xi:
            direct.append(Directions.WEST)
        elif x<xi:
            direct.append(Directions.EAST)
        #y val compare
        if y<yi:
            direct.append(Directions.NORTH)
        elif y>yi:
            direct.append(Directions.SOUTH)
        x = xi
        y = yi
    return direct

def aStarSearch(problem, heuristic):
    def aStarPath():
        def priorityFunction(node):
            state, actions_sequence, path_cost = node
            heuristic_cost = heuristic(state, problem)
            return path_cost+heuristic_cost
        frontier = PriorityQueueWithFunction(priorityFunction)
        return commonSearch(frontier)
    def commonSearch(frontier):
        # as the start location changes every time an barrier is found, using a dynamic start state class variable
        root = problem.dynamicStartState
        explored_set = set()
        actions_sequence = list()
        path_cost = 0
        frontier.push((root, actions_sequence, path_cost))
        while not frontier.isEmpty():
            parent, actions_sequence, path_cost = frontier.pop()
            if parent not in explored_set:
                if problem.getGoalState() == parent:
                    return actions_sequence+[(parent,None)]
                explored_set.add(parent)
                for successor in problem.getSuccessors(parent):
                    state, action, step_cost = successor
                    new_actions_sequence = actions_sequence[:]
                    new_actions_sequence += [(parent, action)]
                    cost = path_cost+step_cost
                    frontier.push((state, new_actions_sequence, cost))

    def planning():
        # generate path
        path = aStarPath()
        if len(path) == 1 and path[0][0] == problem.getGoalState(): 
            return True
        for index in range(len(path)-1):
            currentState, currentAction = path[index]
            nextState, _ = path[index+1]
            problem.finalPath.append((currentState, currentAction))
            if problem.isbarrier(nextState):
                problem.insertbarrier(nextState)
                # update the new start state
                problem.dynamicStartState = currentState
                return False
            elif nextState == problem.getGoalState():
                return True

    def main():
        start_time = datetime.now()
        problem.finalPath = []
        problem.dynamicStartState = problem.getStartState()
        stop = False
        while (problem.dynamicStartState != problem.getGoalState())  and not stop:
            stop = planning()  
        problem.finalPath.append((problem.getGoalState(), None))
        actions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentAction = problem.finalPath[index]
            nextState, _ = problem.finalPath[index+1]
            if currentState != nextState:
                actions.append(currentAction)
                states.append(currentState)
        stop_time = datetime.now()
        time_elapsed = (stop_time - start_time).total_seconds() * 1000
        print('time elapsed: ',time_elapsed)
        print('Size of the Layout: ', str(problem.height)+'x'+str(problem.width))
        print('Path Length: ', len(actions))
        print('Number of barriers: ', len(problem.barriers))
        return actions

    return main()

# The following dstar algorithm is implemented as per the following source
#https://www.aaai.org/Papers/AAAI/2002/AAAI02-072.pdf
def aStarSearchLifeLong(problem, heuristic):
    
    def calculateKey(state):
        g_rhs = min(problem.g[state], problem.rhs[state])
        heuristic_GRHS = g_rhs + heuristic(state, problem)
        return (heuristic_GRHS, g_rhs)
    
    def initialize():
        #Initializing all states as -inf
        for state in problem.getStates():
            problem.rhs[state] = float('inf')
            problem.g[state] = float('inf')
        problem.rhs[problem.dynamicStartState] = 0
        problem.U.insert(problem.dynamicStartState, calculateKey(problem.dynamicStartState))
       
    def updateVertex(u):
        if u != problem.dynamicStartState:
            prevKeys = [float('inf')]
            for nextSuccessor, _, pathCost in problem.getSuccessors(u):
                prevKeys.append(problem.g[nextSuccessor]+pathCost)
            problem.rhs[u] = min(prevKeys)

        problem.U.remove(u)

        if problem.g[u] != problem.rhs[u]:
            problem.U.insert(u, calculateKey(u))
    
    def computeShortestPath():
        goal = problem.getGoalState()
        while (problem.U.topKey() < calculateKey(goal)) or (problem.rhs[goal] != problem.g[goal]):
            u = problem.U.pop()
            if (problem.g[u] > problem.rhs[u]):
                problem.g[u] = problem.rhs[u]                
                for nextSuccessor, _, _ in problem.getSuccessors(u):
                    updateVertex(nextSuccessor)
            else:
                problem.g[u] = float('inf')
                updateVertex(u)
                for nextSuccessor, _, _ in problem.getSuccessors(u):
                    updateVertex(nextSuccessor)

    def planning():
        path = []
        state = (problem.getGoalState(), None)
        path.append(state)

        while state[0] != problem.dynamicStartState:
            minimum = float('inf')
            for successor, action, _ in problem.getSuccessors(state[0]):
                if minimum > problem.g[successor]:
                    minimum = problem.g[successor]
                    state = (successor, Actions.reverseDirection(action))
            path.append(state)

        currentPath = path[::-1]

        if len(currentPath) == 1 and currentPath[0][0] == problem.getGoalState(): 
            return True
            
        for index in range(len(currentPath)-1):
            currentState, currentAction = currentPath[index]
            nextState, _ = currentPath[index+1]
            problem.finalPath.append((currentState, currentAction))
            if problem.isbarrier(nextState):
                problem.insertbarrier(nextState)
                updateVertex(nextState)
                problem.dynamicStartState = currentState
                return False
            elif nextState == problem.getGoalState():
                return True

    def main():
        start_time = datetime.now()
        problem.U = aStarPriorityQueue()
        problem.g = {}
        problem.rhs = {}
        problem.finalPath = []
        problem.dynamicStartState = problem.getStartState()
        initialize()
        stop = False
        while (problem.dynamicStartState != problem.getGoalState())  and not stop:
            initialize()
            computeShortestPath()
            stop = planning()  
        problem.finalPath.append((problem.getGoalState(), None))
        actions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentAction = problem.finalPath[index]
            nextState, _ = problem.finalPath[index+1]
            if currentState != nextState:
                actions.append(currentAction)
                states.append(currentState)
        
        stop_time = datetime.now()
        print(stop_time)
        time_elapsed = (stop_time - start_time).total_seconds() * 1000
        print('time elapsed: ',time_elapsed)
        print('Path Length: ', len(actions))
        print('Size of the Layout: ', str(problem.height)+'x'+str(problem.width))
        print('Number of barriers: ', len(problem.barriers))
        return actions
    return main()

# The following dstar algorithm is implemented as per the following source
#https://www.aaai.org/Papers/AAAI/2002/AAAI02-072.pdf

def dStarSearch(problem):
    def calculateKey(state):
        gRHS = min(problem.g[state], problem.rhs[state])
        distance = problem.k['m']+ gRHS + manhattanDistance(state, problem.s['start'])
        return (distance, gRHS)

    def initialize():
        problem.U = aStarPriorityQueue()
        problem.k = {}
        problem.k['m'] = 0
        problem.g = {}
        problem.rhs = {}
        problem.s = {}
        problem.s['start'] = problem.getStartState()
        problem.s['goal'] = problem.getGoalState()
        # Initializing all states with '-inf' 
        for state in problem.getStates():
            problem.rhs[state] = float('inf')
            problem.g[state] = float('inf')
        problem.rhs[problem.s['goal']] = 0
        problem.U.insert(problem.s['goal'], calculateKey(problem.s['goal']))
      
    def updateVertex(u):
        if u != problem.s['goal']:
            prevKeys = [float('inf')]
            for nextSuccessor, _, pathCost in problem.getSuccessors(u):
                prevKeys.append(problem.g[nextSuccessor]+pathCost)
            problem.rhs[u] = min(prevKeys)
        
        problem.U.remove(u)

        if problem.g[u] != problem.rhs[u]:
            problem.U.insert(u, calculateKey(u))
    
    def computeShortestPath():
        while (problem.U.topKey() < calculateKey(problem.s['start'])) or (problem.rhs[problem.s['start']] != problem.g[problem.s['start']]):
            problem.k['old'] = problem.U.topKey()
            u = problem.U.pop()
            if (problem.k['old'] < calculateKey(u)):
                problem.U.insert(u, calculateKey(u))
            elif (problem.g[u] > problem.rhs[u]):
                problem.g[u] = problem.rhs[u]
                for nextSuccessor, _, _ in problem.getSuccessors(u):
                    updateVertex(nextSuccessor)
            else:
                problem.g[u] = float('inf')
                updateVertex(u)
                for nextSuccessor, _, _ in problem.getSuccessors(u):
                    updateVertex(nextSuccessor)

    def main():
        start_time = datetime.now()
        initialize()
        computeShortestPath()
        problem.finalPath = []
        problem.s['last'] = problem.s['start']
        problem.dynamicAction = None
        while (problem.s['start'] != problem.s['goal']):
            if problem.g[problem.s['start']] == float('inf'):
                return []
            minimum = float('inf')
            problem.s['successor'] = None

            for nextSuccessor, action, pathCost in problem.getSuccessors(problem.s['start']):
                updatedPathCost = problem.g[nextSuccessor]+pathCost
                if updatedPathCost < minimum:
                    minimum = updatedPathCost
                    problem.s['successor'] = nextSuccessor
                    problem.dynamicAction = action
                
            if problem.isbarrier(problem.s['successor']):
                problem.insertbarrier(problem.s['successor'])
                problem.k['m'] = problem.k['m']+ manhattanDistance(problem.s['last'], problem.s['start'])
                problem.s['last'] = problem.s['start']
                updateVertex(problem.s['successor'])
                computeShortestPath()
            else:
                problem.finalPath.append((problem.s['start'], problem.dynamicAction))
                problem.s['start'] = problem.s['successor']

        problem.finalPath.append((problem.s['goal'], None))
        actions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentAction = problem.finalPath[index]
            nextState, _ = problem.finalPath[index+1]
            if currentState != nextState:
                actions.append(currentAction)
                states.append(currentState)

        stop_time = datetime.now()
        time_elapsed = (stop_time - start_time).total_seconds() * 1000
        print('time elapsed: ',time_elapsed)
        print('Size of the Layout: ', str(problem.height)+'x'+str(problem.width))
        print('Path Length: ', len(actions))
        print('Number of barriers: ', len(problem.barriers))
        return actions

    return main()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
astar = aStarSearch
astarll = aStarSearchLifeLong
dstar = dStarSearch