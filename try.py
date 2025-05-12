import json
import numpy as np
import torch
from opposetbuild import PokerSACAgent

class PokerStateEncoder:
    def __init__(self, player_seat_id=3):
        self.player_seat_id = player_seat_id  # 需要监控的玩家座位ID
        self.max_chips = 2000  # 归一化参考值
        self.max_public_cards = 5  # 最多5张公共牌
        
        # 映射表初始化
        self.rank_map = {'2':0, '3':1, '4':2, '5':3, '6':4,
                        '7':5, '8':6, '9':7, 'T':8, 'J':9,
                        'Q':10, 'K':11, 'A':12}
        self.suit_map = {'c':0, 'd':1, 'h':2, 's':3}
        self.stage_map = {
            "PREFLOP": 0,
            "FLOP": 1,
            "TURN": 2,
            "RIVER": 3,
            "SHOWDOWN": 4
        }

    def _parse_card(self, card):
        """解析单张牌的数字或字符串表示"""
        if isinstance(card, int):  # 处理数值型表示
            rank = card // 13
            suit = card % 13
            return [rank, suit] if suit < 4 else [rank, suit-13]  # 修正错误编码
        elif isinstance(card, str):  # 处理字符串表示 "3c"
            return [self.rank_map[card[0]], self.suit_map[card[1]]]
        return [0, 0]  # 空牌填充

    def _get_current_stage(self, data):
        """获取当前游戏阶段"""
        # 从最后一条非SHOWDOWN的记录获取当前阶段
        for record in reversed(data["dynamic_info"]["round_history"]):
            if record["stage"] != "SHOWDOWN":
                return record["stage"]
        return "PREFLOP"

    def encode(self, json_data):
        # 加载JSON数据
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        
        # 初始化特征容器
        features = []

        # 1. 手牌特征 --------------------------------------------------
        player_hand = []
        for record in data["dynamic_info"]["round_history"]:
            if record["player"] == self.player_seat_id and record["hand_cards"]:
                player_hand = record["hand_cards"]
                break
        
        # 解析手牌 (2张)
        hand_feature = []
        for card in player_hand[:2]:  # 只取前两张
            parsed = self._parse_card(card)
            hand_feature.extend(parsed)
        # 填充不足的牌
        hand_feature += [0, 0]*(2 - len(player_hand))
        features.extend(hand_feature)

        # 2. 公共牌特征 -----------------------------------------------
        table_cards = []
        current_stage = self._get_current_stage(data)
        # 找到最新有效的公共牌记录
        for record in reversed(data["dynamic_info"]["round_history"]):
            if record["table_cards"] and isinstance(record["table_cards"][0], (int, str)):
                table_cards = record["table_cards"]
                break
        
        # 解析公共牌 (最多5张)
        public_feature = []
        for card in table_cards[:self.max_public_cards]:
            parsed = self._parse_card(card)
            public_feature.extend(parsed)
        # 填充不足的牌
        public_feature += [0, 0]*(self.max_public_cards - len(table_cards))
        features.extend(public_feature)

        # 3. 筹码特征 -------------------------------------------------
        player_chips = next((s["hand_chips"] for s in data["basic_info"]["seat_info"] 
                          if s["seatid"] == self.player_seat_id), 0)
        # opponent_seat = 4 if self.player_seat_id == 3 else 3
        opponent_chips = next((s["hand_chips"] for s in data["basic_info"]["seat_info"] 
                             if s["seatid"] != self.player_seat_id), 0)
        
        # 计算底池
        pot = sum([r.get("bet",0) for r in data["dynamic_info"]["round_history"]])
        
        # 归一化处理
        chip_features = [
            player_chips / self.max_chips,
            opponent_chips / self.max_chips,
            pot / self.max_chips
        ]
        features.extend(chip_features)

        # 4. 位置特征 -------------------------------------------------
        position = [0, 0, 0]  # [庄家, 小盲, 大盲]
        dealer_id = data["basic_info"]["dealer_info"]["seatid"]
        if self.player_seat_id == dealer_id:
            position[0] = 1
        if self.player_seat_id == data["basic_info"]["blind"]["small_blind"]["seatid"]:
            position[1] = 1
        if self.player_seat_id == data["basic_info"]["blind"]["big_blind"]["seatid"]:
            position[2] = 1
        features.extend(position)

        # 5. 阶段特征 -------------------------------------------------
        stage_onehot = [0]*5
        stage_onehot[self.stage_map[current_stage]] = 1
        features.extend(stage_onehot)

        # 6. 对手行为特征 ---------------------------------------------
        opponent_actions = []
        for r in data["dynamic_info"]["round_history"]:
            if r["player"] != self.player_seat_id:
                opponent_actions.append(r.get("type", 0))
        
        # 统计动作类型：0-弃牌，1-跟注，2-加注
        action_counts = [0, 0, 0]
        for a in opponent_actions[-10:]:  # 只看最近10个动作
            if 0 <= a <= 2:
                action_counts[a] += 1
        total = len(opponent_actions) or 1
        aggression = action_counts[2] / total  # 加注频率
        features.append(aggression)

        # 转换为numpy数组并确保维度
        state_vector = np.array(features, dtype=np.float32)
        
        # 最终维度验证
        expected_dim = 2*2 + 5*2 + 3 + 3 + 5 + 1
        assert len(state_vector) == expected_dim, \
            f"维度错误: 预期{expected_dim}, 实际{len(state_vector)}"
        
        return state_vector

def load_json_file(file_path):
    data = []
    with open(file_path, "r") as f:
        content = f.read()
    
    # 假设每个 JSON 对象之间有一个空行作为分隔符
    json_objects = content.split("\n\n")
    
    for json_str in json_objects:
        try:
            # 尝试解析每个 JSON 对象
            json_obj = json.loads(json_str)
            data.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            continue
    
    return data


# 测试用例 --------------------------------------------------------
if __name__ == "__main__":
    # 加载示例数据
    # with open("log/cAMPSRjo/log20250510_164759.json") as f:
    #     sample_data = load_json_file(f)
    sample_data = load_json_file("log/cAMPSRjo/log20250510_164759.json")
    sample_data = sample_data[1]
    
    encoder = PokerStateEncoder(player_seat_id=3)
    
    # 执行编码
    state = encoder.encode(sample_data)
    state = torch.tensor(state)
    print("编码后的状态向量:")
    print(type(state))
    print(f"向量维度: {len(state)}")
    
    # 维度验证
    assert len(state) == (4 + 10 + 3 + 3 + 5 + 1), "特征维度不匹配"