import math
import random
from agents.base_agent import BaseAgent

class MCTSNode:
    """
    Node in the MCTS tree.
    Each node represents a state and tracks statistics for UCB-based selection.
    """
    def __init__(self, untried_moves, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried = list(untried_moves)

    def is_fully_expanded(self):
        """Returns True if all possible moves from this node have been tried."""
        return len(self.untried) == 0

    def uct_score(self, exploration=1.41):
        """
        Calculate the UCT (Upper Confidence bound applied to Trees) score.
        UCT balances exploitation (winning rate) and exploration (visit count).
        """
        if self.visits == 0:
            return math.inf
        
        # Avoid calling uct_score on root (parent=None)
        if self.parent is None:
            return 0
        
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term

    def best_child(self, exploration=1.41):
        """Return child node with the highest UCT score."""
        return max(self.children, key=lambda s: s.uct_score(exploration))

    def expand_node(self):
        """
        Expansion phase: create a new child node by selecting an untried move.
        """
        move = random.choice(self.untried)
        self.untried.remove(move)
        child = MCTSNode([], parent=self, action=move)
        self.children.append(child)
        return child





class MCTS_Agent(BaseAgent):
    """
    Monte-Carlo Tree Search agent for Texas Hold'em poker.
    
    MCTS phases:
    1. Selection: Traverse tree using UCT until reaching an expandable node
    2. Expansion: Create a new child node for an untried action
    3. Simulation: Evaluate the node using a rollout policy (hand strength heuristic)
    4. Backpropagation: Update statistics back up the tree
    """
    
    def __init__(self, name="MCTS_Agent", num_simulations=500):
        super().__init__(name)
        self.number_of_simulations = num_simulations

    def estimate_hand_strength(self, observation):
        if isinstance(observation, dict):
            observation = observation.get("observation", observation)

        if len(observation) == 0:
            return 0.5

        # Actual layout from PettingZoo (54 elements total):
        # [0:52]  → one-hot card vector (hole + community cards combined)
        # [52]    → normalized pot size
        # [53]    → normalized call amount / chip count

        card_vec  = observation[:52]
        pot_size  = observation[52]
        call_amt  = observation[53]

        # Count active cards — max is 7 (2 hole + 5 community)
        num_cards = sum(card_vec)
        card_strength = min(1.0, num_cards / 7.0)

        # Factor in pot odds — if call amount is high relative
        # to pot, we need a stronger hand to justify calling
        if pot_size > 0:
            pot_odds = call_amt / (pot_size + call_amt + 1e-6)
        else:
            pot_odds = 0.0

        # Combine card strength with pot odds adjustment
        # Higher pot odds = need stronger hand to call profitably
        strength = card_strength * (1.0 - pot_odds * 0.3)

        return max(0.0, min(1.0, strength))
    
    def act(self, observation, action_mask):
        """
        Main decision function: use MCTS to select the best action.
        
        Args:
            observation: Current game state observation
            action_mask: Binary array indicating legal actions
            
        Returns:
            int: Action index (0=fold, 1=call, 2-4=raise variants)
        """
        # Get all legal moves
        legal_moves = [i for i, legal in enumerate(action_mask) if legal]

        # If only one move is legal, take it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Estimate hand strength for use in simulations
        hand_strength = self.estimate_hand_strength(observation)

        # Create root node with all legal moves as untried
        root = MCTSNode(legal_moves)

        # Run MCTS iterations
        for _ in range(self.number_of_simulations):
            # Selection & Expansion
            node = self._mcts_select_and_expand(root)
            
            # Simulation
            reward = self._mcts_simulate(node, hand_strength, legal_moves)
            
            # Backpropagation
            self._mcts_backpropagate(node, reward)
        
        # Return the action that was visited the most (most promising)
        if not root.children:
            return random.choice(legal_moves)
        
        best_child = max(root.children, key=lambda r: r.visits)
        return best_child.action
    
    def _mcts_select_and_expand(self, root):
        """
        Selection & Expansion phases combined:
        - Selection: Traverse tree using UCT until reaching a non-fully-expanded node
        - Expansion: Create a new child if node is not fully expanded
        
        Returns: The node to simulate from
        """
        node = root
        
        # Selection: descend tree using UCT
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        
        # Expansion: if node has untried moves, create a new child
        if not node.is_fully_expanded():
            node = node.expand_node()
        
        return node
    
    def _mcts_simulate(self, node, hand_strength, legal_moves, max_depth=10):
        current_strength = hand_strength
        depth = 0

        while depth < max_depth:
            action = self._sample_rollout_action(legal_moves, current_strength)
            depth += 1

            # Terminal condition — fold ends the hand
            if action == 0:
                return 1.0 - current_strength

            # Decay strength slightly each step to simulate
            # uncertainty of future community cards
            noise = random.uniform(-0.05, 0.05)
            current_strength = max(0.0, min(1.0, current_strength + noise))

        return self._evaluate_action(node.action, current_strength)

    def _sample_rollout_action(self, legal_moves, hand_strength):
        weights = [self._action_weight(a, hand_strength) for a in legal_moves]
        total = sum(weights)
        if total == 0:
            return random.choice(legal_moves)
        normalized = [w / total for w in weights]
        return random.choices(legal_moves, weights=normalized, k=1)[0]
    
    def _evaluate_action(self, action, hand_strength):
        """
        Heuristically evaluate the quality of an action given current hand strength.
        
        Actions:
            0 = Fold: Give up the current pot. Bad when hand is strong.
            1 = Call: Match current bet. Neutral option.
            2 = Raise 50%: Moderate bet increase. Better with stronger hands.
            3 = Raise 100%: Aggressive bet. Good with very strong hands.
            4 = Raise All-In: Maximum risk/reward. Depends on variance.
        
        Returns: float between 0 and 1
        """
        if action == 0:  # Fold
            # Folding is bad when you have a strong hand
            return 1.0 - hand_strength
        
        elif action == 1:  # Call
            # Calling is a neutral, passive option
            return 0.5
        
        elif action == 2:  # Raise 50% of pot
            # Small raise - slightly better with strong hands
            return 0.55 + (hand_strength * 0.15)
        
        elif action == 3:  # Raise 100% of pot
            # Larger raise - significantly better with strong hands
            return 0.45 + (hand_strength * 0.40)
        
        elif action == 4:  # All-In
            return hand_strength ** 2  # Only good with very strong hands
                
        else:
            return 0.5
    
    def _action_weight(self, action, hand_strength):
        """Alias for _evaluate_action, used during rollout sampling."""
        return self._evaluate_action(action, hand_strength)
    
    def _mcts_backpropagate(self, node, reward):
        """
        Backpropagation phase: Update node statistics from leaf back to root.
        
        Updates visits and wins for all ancestors of the node.
        """
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

    def reset(self):
        """Called at the start of each new game."""
        pass

    def record_action(self, action):
        """Override if agent needs to track action history."""
        pass  # Not needed for basic MCTS