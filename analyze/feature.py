import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple


class PokerFeatureEngine:
    def __init__(self, big_blind: int = 20):
        self.big_blind = big_blind
        self.history_window = 20  # 保留最近20手的记忆
        self.opponent_model = OpponentRangeEstimator()
        self.hand_strength_calc = HandStrengthCalculator()
        
        # 初始化历史记忆缓存
        self.action_history = deque(maxlen=self.history_window)
        self.pot_history = deque(maxlen=self.history_window)
        self.showdown_history = deque(maxlen=self.history_window)

    def _encode_card(self, card_id: int) -> str:
        """将卡牌ID转换为标准扑克表示法"""
        suits = ['s', 'd', 'h', 'c']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suit = suits[card_id // 13]
        rank = ranks[card_id % 13]
        return rank + suit

    def _get_position(self, seat_info: dict, player_id: str) -> str:
        """确定玩家位置"""
        seats = sorted(seat_info, key=lambda x: x['seatid'])
        dealer_id = seat_info[0]['seatid']  # 假设第一个座位是庄家
        positions = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
        return positions[(seats.index(player_id) - dealer_id) % len(seats)]

    def _calculate_pot_odds(self, current_bet: float, pot_size: float) -> float:
        """计算底池赔率"""
        return current_bet / (pot_size + current_bet) if pot_size > 0 else 0

    def _extract_action_features(self, action_sequence: List[dict]) -> Dict:
        """编码动作序列特征"""
        action_mapping = {
            2: 'bet',
            5: 'fold',
            # 添加其他动作类型映射
        }
        
        encoded = {
            'action_types': [],
            'bet_sizes': [],
            'timing_stats': defaultdict(float)
        }
        
        for idx, action in enumerate(action_sequence):
            action_type = action_mapping.get(action['type'], 'unknown')
            encoded['action_types'].append(action_type)
            
            # 标准化下注量
            if action_type in ['bet', 'raise']:
                bb_ratio = action['bet'] / self.big_blind
                encoded['bet_sizes'].append(bb_ratio)
                
            # 时间特征（假设有action_time字段）
            if 'action_time' in action:
                encoded['timing_stats']['mean_time'] += action['action_time']
                if idx > 0:
                    encoded['timing_stats']['time_variance'] += (action['action_time'] - prev_time)**2
                prev_time = action['action_time']
        
        # 后处理统计量
        if encoded['bet_sizes']:
            encoded['bet_size_stats'] = {
                'mean': np.mean(encoded['bet_sizes']),
                'std': np.std(encoded['bet_sizes']),
                'last': encoded['bet_sizes'][-1]
            }
        return encoded

    def _analyze_board_texture(self, community_cards: List[int]) -> Dict:
        """分析公共牌面特征"""
        cards = [self._encode_card(c) for c in community_cards if c != -1]
        suits = [c[1] for c in cards]
        ranks = [RANKS[c[0]] for c in cards]
        
        return {
            'flush_possible': len(set(suits)) < len(cards),
            'straight_possible': self._check_straight_possible(ranks),
            'paired': len(ranks) != len(set(ranks)),
            'high_card': max(ranks) if ranks else 0,
            'coordination': len([r for r in ranks if r >= 10]) / len(cards) if cards else 0
        }

    def _check_straight_possible(self, ranks: List[int]) -> bool:
        """检查顺子可能性"""
        unique_ranks = sorted(list(set(ranks)))
        if len(unique_ranks) < 3:
            return False
        for i in range(len(unique_ranks)-4):
            if unique_ranks[i+4] - unique_ranks[i] == 4:
                return True
        return False

    def build_features(self, raw_data: dict) -> Dict:
        """主特征构建方法"""
        # 基础信息处理
        seat_info = raw_data['basic_info']['seat_info']
        hero_info = next(s for s in seat_info if s['usrname'] == 'p_13304936695')
        villain_info = next(s for s in seat_info if s['usrname'] != 'p_13304936695')
        
        # 位置特征
        features = {
            'position': {
                'hero': self._get_position(seat_info, hero_info['seatid']),
                'villain': self._get_position(seat_info, villain_info['seatid'])
            },
            'stack_ratio': hero_info['hand_chips'] / villain_info['hand_chips'],
            'effective_bb': min(hero_info['hand_chips'], villain_info['hand_chips']) / self.big_blind
        }
        
        # 动作序列分析
        action_features = self._extract_action_features(raw_data['dynamic_info']['round_history'])
        features.update(action_features)
        
        # 牌面分析
        community_cards = [c for c in raw_data['dynamic_info']['round_history'][-1]['table_cards'] if c != -1]
        features['board'] = self._analyze_board_texture(community_cards)
        
        # 范围估计
        features['range'] = self.opponent_model.estimate_range(
            action_sequence=action_features['action_types'],
            board_texture=features['board']
        )
        
        # 手牌强度
        hero_hand = [self._encode_card(c) for c in raw_data['dynamic_info']['round_history'][0]['hand_cards']]
        features['hand_strength'] = self.hand_strength_calc.calculate(
            hole_cards=hero_hand,
            community_cards=community_cards,
            opponent_range=features['range']
        )
        
        # 动态记忆特征
        features['historical'] = {
            'aggression': self._calculate_aggression_factor(),
            'continuation_bet': self._calculate_cbet_rate(),
            'win_rate': np.mean(self.showdown_history) if self.showdown_history else 0.5
        }
        
        return features

    def _calculate_aggression_factor(self) -> float:
        """计算对手攻击指数"""
        agg_actions = sum(1 for a in self.action_history if a in ['bet', 'raise'])
        return agg_actions / len(self.action_history) if self.action_history else 0.5

    def _calculate_cbet_rate(self) -> float:
        """计算持续下注率"""
        if len(self.pot_history) < 2:
            return 0.5
        cbet_count = sum(1 for i in range(1, len(self.pot_history)) 
                      if self.pot_history[i] > self.pot_history[i-1])
        return cbet_count / (len(self.pot_history)-1)