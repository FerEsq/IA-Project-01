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

'''
 * Nombre: search.py
 * Programadores: Fernanda Esquivel, Elías Alvarado, Adrian Fulladolsa
 * Lenguaje: Python
 * Historial: Modificado el 27.03.2024
'''

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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
    """

    stack = util.Stack()
    stack.push((problem.getStartState(), []))
    
    #Inicializar el conjunto de estados visitados para llevar un registro de los estados ya explorados
    visited = set()

    while not stack.isEmpty():
        #Sacar el último elemento añadido al stack
        state, actions = stack.pop()

        #Si el estado es el goal state, devolver el camino que llevó al pacman hasta aquí
        if problem.isGoalState(state):
            return actions

        #Si el estado no ha sido visitado, marcarlo como visitado
        if state not in visited:
            visited.add(state)

            #Obtener los sucesores del estado actual y añadirlos al stack
            for nextState, action, _ in problem.expand(state):
                #Añadir el sucesor al stack solo si no ha sido visitado
                if nextState not in visited:
                    #Añadir el nuevo estado y la lista de acciones actualizada al stack
                    stack.push((nextState, actions + [action]))

    #Si el stack se vacía sin encontrar el objetivo, devolver una lista vacía
    return []

    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    #Queue para los nodos por explorar
    frontier = util.Queue()
    frontier.push((problem.getStartState(), [])) #Se inicia con el estado inicial y un camino vacío

    #Conjunto almacenar los nodos ya visitados
    explored = set()

    while not frontier.isEmpty():
        #Se saca el nodo actual y el camino para llegar allí de la queue
        state, actions = frontier.pop()

        #Si el nodo actual es un goal state, devolver el camino que llevó al pacman hasta aquí
        if problem.isGoalState(state):
            return actions

        #Si el nodo actual no ha sido visitado, se marca como explorado
        if state not in explored:
            explored.add(state)

            #Para cada sucesor del nodo actual, si no ha sido explorado, lo añadimos a la frontera
            for nextState, action, _ in problem.expand(state):
                if nextState not in explored:
                    #Añadimos el sucesor a la frontera con el camino actualizado para llegar allí
                    frontier.push((nextState, actions + [action]))

    #Si la frontera está vacía y no hemos encontrado un estado objetivo, devolvemos una lista vacía
    return []

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue

    #PriorityQueue para almacenar los nodos junto con su coste total (coste hasta ahora + heurístico) y ordenarlos
    openList = PriorityQueue()
    
    #El estado inicial se añade a la cola con un coste de 0 + heurístico
    startNode = problem.getStartState()
    openList.push((startNode, [], 0), 0 + heuristic(startNode, problem))
    
    #Visited se utiliza para almacenar los nodos ya evaluados
    visited = set()
    
    while not openList.isEmpty():
        #Extrae el nodo con el menor coste total (coste hasta ahora + heurístico)
        currentNode, actions, currentCost = openList.pop()
        
        #Si el nodo actual es un goal state, devolver el camino que llevó al pacman hasta aquí
        if problem.isGoalState(currentNode):
            return actions
        
        #Si el nodo no ha sido visitado, se processa
        if currentNode not in visited:
            visited.add(currentNode)
            
            #Expanción del nodo actual
            for nextState, action, stepCost in problem.expand(currentNode):
                if nextState not in visited:
                    #El nuevo coste es el coste actual más el coste de dar el siguiente paso
                    newCost = currentCost + stepCost
                    #Añade el sucesor a la cola con el nuevo coste y la heurística
                    openList.push((nextState, actions + [action], newCost), newCost + heuristic(nextState, problem))
    
    #Si la cola se vacía sin encontrar una solución, eleva una excepción
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
