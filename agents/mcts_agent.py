import math
import random
import time
import game.environment as environment

class MCTSNode:
    def __init__(self, untried_moves, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried = list(untried_moves)

    #Returns true if node has no children, false otherwise
    def is_fully_expanded(self):
        return len(self.untried) == 0

    #Calculates and returns the UCT score for a given node.
    def uct_score(self, exploration=1.41):
        if self.visits == 0:
            return math.inf
        else:
            return (self.wins / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits))

    #Returns child node with the highest UCT score
    def best_child(self, exploration=1.41):
        return max(self.children, key=lambda s: s.uct_score(exploration))

    #Returns resulting state from applying a move to given state
    def expand_node(self):
        move = random.choice(self.untried)
        self.untried.remove(move)
        #Create Child Node
        child = MCTSNode([], parent = self, action = move)
        self.children.append(child)
        return child



    
class MCTS_Agent:
    def __init__(self, name = "MCTS_Agent"):
        self.name = "MCTS_Agent"
        self.number_of_simulations = 500

    def estimate_hand_strength(self, observation):
        obs = observation
        if isinstance(obs, dict):
            obs = obs.get("observation", obs)
        num_of_ones = sum(obs)
        return num_of_ones / len(obs)
    
    def act(self, observation, action_mask):
        #Get all legal moves for current state
        legal_moves = [i for i, legal in enumerate(action_mask) if legal]

        #If there is only one legal move, return it
        if len(legal_moves) == 1:
            return legal_moves[0]

        win_probability = self.estimate_hand_strength(observation)

        root = MCTSNode(legal_moves)

        #Run MCTS for all legal moves
        for _ in range(self.number_of_simulations):
            #Selection
            node = self.mcts_select(root)
            #Expansion
            if not node.is_fully_expanded():
                node = node.expand_node()
            #Simulation
            reward = self.mcts_simulate(node, win_probability, legal_moves)
            #Backprop
            self.mcts_backpropagate(node, reward)
        
        #Return Best Choice
        return max(root.children, key=lambda r: r.visits).action
    
    #Selection Phase
    def mcts_select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node
    
    #Simulation Phase
    def mcts_simulate(self, node, win_probability, legal_moves):
        if node.action is not None:
            action = node.action
        else:
            action = random.choice(legal_moves)
        
        action_bonus = self.action_quality(action, win_probability)
        return win_probability + action_bonus
    
    def action_quality(self, action, win_probability):
        #0 = Fold
        #1 = Call
        #2 = Raise 50% Pot
        #3 = Raise 100% Pot
        #4 = Raise All-In
        if action == 0:
            return 1 - win_probability
        elif action == 1:
            return 0.5
        elif action == 2:
            return win_probability
        elif action == 3:
            return win_probability
        elif action == 4:
            return win_probability
        else:
            return 0.5
    
    #Backpropogation Phase
    def mcts_backpropagate(self, node, reward):
        while node != None:
            node.visits += 1
            node.wins += reward
            node = node.parent
        return

    def reset(self):
        pass

    def record_action(self, action):
        pass  # Override if agent tracks action history