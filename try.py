import copy
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Union  # 修改1：添加Any类型
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Optional

import joblib
manager = mp.Manager()
class Player:
    def __init__(self, hand: List[str], chips: int):
        self.hand = hand
        self.chips = chips
        self.current_bet = 0
        self.is_active = True

class Agent:
    """
    Create agent, optionally initialise to agent specified at path.

    ...

    Attributes
    ----------
    strategy : Dict[str, Dict[str, int]]
        The preflop strategy for an agent.
    regret : Dict[str, Dict[strategy, int]]
        The regret for an agent.
    """
    def __init__(
        self,
        agent_path: Optional[Union[str, Path]] = None,
        use_manager: bool = True,
    ):
        """Construct an agent."""
        # Don't use manager if we are running tests.
        testing_suite = bool(os.environ.get("TESTING_SUITE", False))
        use_manager = use_manager and not testing_suite
        dict_constructor: Callable = manager.dict if use_manager else dict
        self.strategy = dict_constructor()
        self.regret = dict_constructor()
        if agent_path is not None:
            saved_agent = joblib.load(agent_path)
            # Assign keys manually because I don't trust the manager proxy.
            for info_set, value in saved_agent["regret"].items():
                self.regret[info_set] = value
            for info_set, value in saved_agent["strategy"].items():
                self.strategy[info_set] = value


class PokerState:
    def __init__(self, players: List[Player], board: List[str], pot: int, current_player: int):
        self.player_i = 
        self.players = players
        self.is_terminal = False
        
        self.info_set = 
        self.inital_regret = 0
        
        self.payout = 
        self.legal_actions = ["FOLD", "CHECK", "CALL", "BET"]
        self.inital_regret = 
    
    def get_stage(self, stage:int):
        if stage == "SHUTDOWN":
            self.is_terminal = True
            
            
    def apply_action(self, action:str):
        
    
    
        
        

def calculate_strategy(this_info_sets_regret: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the strategy based on the current information sets regret.

    ...

    Parameters
    ----------
    this_info_sets_regret : Dict[str, float]
        Regret for each action at this info set.

    Returns
    -------
    strategy : Dict[str, float]
        Strategy as a probability over actions.
    """
    # TODO: Could we instanciate a state object from an info set?
    actions = this_info_sets_regret.keys()
    regret_sum = sum([max(regret, 0) for regret in this_info_sets_regret.values()])
    if regret_sum > 0:
        strategy: Dict[str, float] = {
            action: max(this_info_sets_regret[action], 0) / regret_sum
            for action in actions
        }
    else:
        default_probability = 1 / len(actions)
        strategy: Dict[str, float] = {action: default_probability for action in actions}
    return strategy



def cfrp(
    agent: Agent,
    state: PokerState,
    i: int,
    t: int,
    c: int,
    locks: Dict[str, mp.synchronize.Lock] = {},
):
    """
    Counter factual regret minimazation with pruning.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The Player.
    t : int
        The iteration.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]

    elif ph == i:
        # calculate strategy
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        vo = 0.0
        voa: Dict[str, float] = dict()
        # Explored dictionary to keep track of regret updates that can be
        # skipped.
        explored: Dict[str, bool] = {action: False for action in state.legal_actions}
        # Get the regret for this state.
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_actions:
            if this_info_sets_regret[action] > c:
                new_state: PokerState = state.apply_action(action)
                voa[action] = cfrp(agent, new_state, i, t, c, locks)
                explored[action] = True
                vo += sigma[action] * voa[action]
        if locks:
            locks["regret"].acquire()
        # Get the regret for this state again, incase any other process updated
        # it whilst we were doing `cfrp`.
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_actions:
            if explored[action]:
                this_info_sets_regret[action] += voa[action] - vo
        # Update the master copy of the regret.
        agent.regret[state.info_set] = this_info_sets_regret
        if locks:
            locks["regret"].release()
        return vo
    else:
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        new_state: PokerState = state.apply_action(action)
        return cfrp(agent, new_state, i, t, c, locks)

if __name__ == "__main__":
    # 测试案例
    player0 = Player(hand=['Ah', 'Kh'], chips=1000)
    player1 = Player(hand=['Qc', 'Jc'], chips=1000)
    initial_state = PokerState(
        players=[player0, player1],
        board=[],
        pot=0,
        current_player=0
    )
    
    agent = Agent()
    
    # 简单训练
    for t in range(1, 3):  # 减少迭代次数方便测试
        for i in range(2):
            cfrp(agent, initial_state, i, t, c=0)
    
    # 输出结果
    print("训练结果示例:")
    for info_set, regret in agent.regret.items():
        print(f"信息集: {info_set}")
        strategy = calculate_strategy(regret)
        for action, prob in strategy.items():
            print(f"  {action}: {prob:.2%}")