import pandas as pd
from constant_rate import win_rate, win_rate_heads_up, RANKS
from collections import Counter
import argparse
import json
import os
from log_processor import RoundInfoManager
from opposetbuild import OpponentAnalysis
# 通过牌面好坏的对照表计算惩罚率
card_sequence = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

round_name = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
# 把比赛结果放到一个链表中，根据最后一个节点的值反向递归计算出每个状态的标签值，标签值我们只关注最后的输赢
    

class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = None
        self.value = 0
        
    """ 
    batch = {
        "stage":"PREfLOP",
        "hand_cards":["Ah", "As"],
        "public_cards":...
        "action":"Bet",
        "history_action":[[a1, a2, a3, a4, a5],[...]],
        "bet_history":[[20, 10, 20, 30, 50], [1900, 20, 30, 20, 10]],
        "hand_chips":2000
    }
    """
    def load_info(self, batch:dict):
        self.stage = batch["stage"]
        self.private_cards = batch["hand_cards"]
        self.public_cards = batch["public_cards"]
        self.action = batch["action"]
        self.act_his = batch["history_action"]
        self.bet_his = batch["bet_history"]
        self.hand_chips = batch["hand_chips"]
        self.totalbet = batch["totalbet"]
        self.player = batch["player"]
        self.location = batch["location"]
        self.oppo_totalbet = batch["oppo_totalbet"]
        self.aggression = batch["aggression"]
        self.oppo_fold = batch["oppo_fold"]
    
    def calculate_value(self, baseline:float, c_puct:float):
        # 仅适用于叶子节点，baseline是基础奖罚值，c_puct是根据手牌获得的倍率
        if self.children != None:
            raise ValueError("node is not leaf")
        else:
            if baseline > 0:
                plus_param = 100-c_puct
            else:
                plus_param = c_puct
            self.value = baseline*plus_param
    
    def calculate_recursive(self, gamma:float):
        # 非叶子节点
        if self.children == None:
            raise ValueError("node is leaf")
        else:
            # 胜负向上平减
            self.value = self.children.value*gamma
            
    def add_child(self, child):
        self.children = child
        child.parent = self
        
        
class node_list:
    def __init__(self, gamma:float=0.9, baseline:float=1):
        self.gamma = gamma
        self.baseline = baseline
        self.root = Node()
        self.node_now = self.root
        self.node_counter = 1
        self.winner = None
        
    def add_node(self, node:Node, batch:dict):
        if not self.winner:
            self.winner = batch["winner"]
        node.load_info(batch)
        self.node_now.add_child(node)
        self.node_now = node
        self.node_counter += 1
        # print(f"Node number is {self.node_counter}")
    
    def backbone(self, c_puct:float):
        # 从叶节点开始计算
        base_line = self.baseline*self.winner
        while self.node_now.children != None:
            self.node_now = self.node_now.children
        self.node_now.calculate_value(base_line, c_puct)
        # self.node_now = self.node_now.parent
        # 一直计算到root
        while self.node_now.parent != None:
            self.node_now = self.node_now.parent
            self.node_now.calculate_recursive(self.gamma)
            
    def report_info(self):
        result = {
            "stage": [],
            "private_cards": [],
            "public_cards": [],
            "action": [],
            "history_action": [],
            "bet_history": [],
            "hand_chips": [],
            "value": [],
            "totalbet": [],
            "player":  [],
            "location": [],
            "oppo_totalbet": [],
            "aggression": [],
            "oppo_fold": []
        }
        
        current_node = self.root.children
        while current_node is not None and hasattr(current_node, 'stage'):  # 遍历有效节点
            result["stage"].append(current_node.stage)
            result["private_cards"].append(current_node.private_cards)
            result["public_cards"].append(current_node.public_cards)
            result["action"].append(current_node.action)
            result["history_action"].append(current_node.act_his)
            result["bet_history"].append(current_node.bet_his)
            # print(result["bet_history"])
            result["hand_chips"].append(current_node.hand_chips)
            result["value"].append(current_node.value)
            result["totalbet"].append(current_node.totalbet)
            result["player"].append(current_node.player)
            result["location"].append(current_node.location)
            result["oppo_totalbet"].append(current_node.oppo_totalbet)
            result["aggression"].append(current_node.aggression)
            result["oppo_fold"].append(current_node.oppo_fold)

            current_node = current_node.children
        return result

class LabelScorer:
    def __init__(self):
        self.result = {
            "stage":[],
            "private_cards":[],
            "public_cards":[],
            "action":[],
            # 每一个元素都是一个列表，记录了牌桌上前两回合所有玩家的出牌情况
            "history_action":[],
            # 每一个元素也是一个列表，记录前两回合玩家押注情况
            "bet_history":[],
            # 我的筹码数量
            "hand_chips":[],
            "totalbet": [],
            "player": [],
            "location": [],
            "oppo_totalbet": [],
            "aggression": [],
            "oppo_fold": []
        }
        
        self.path = "data.csv"

        self.counter = 0
        
    def get_result(self, node_list:node_list):
        node_data = node_list.report_info()
        for key in self.result:
            self.result[key].extend(node_data[key])
        if not hasattr(self, 'value_list'):
            self.value_list = []
        self.value_list.extend(node_data["value"])
        self.counter += len(node_data["stage"])
        
    
    def report_all_result(self):
        prompts = []
        for i in range(len(self.result["stage"])):
            # 格式化历史动作
            formatted_actions = " | ".join(
                [",".join(round_actions) for round_actions in self.result["history_action"][i]]
            )
            # print(self.result["bet_history"])
            # 格式化押注记录
            formatted_bets = " | ".join(
                [f"Round{idx+1}:"+",".join(map(str,bets)) 
                 for idx, bets in enumerate(self.result["bet_history"][i])]
            )
            # 构建描述字符串
            desc = (
                f"Stage: {self.result['stage'][i].lower()}, "
                f"Private Cards: {', '.join(str(card) for card in self.result['private_cards'][i]) if self.result['private_cards'][i] else 'None'}, "
                f"Public Cards: {', '.join(str(card) for card in self.result['public_cards'][i]) if self.result['public_cards'][i] else 'None'}, "
                f"Action: {self.result['action'][i]}, "
                f"History Actions: [{formatted_actions}], "
                f"Bet History: [{formatted_bets}], "
                f"Chips: {self.result['hand_chips'][i]}, "
                f"Totalbet: {self.result['totalbet'][i]}, "
                f"Player: {self.result['player'][i]}, "
                f"Location: {self.result['location'][i]}, "
                f"Oppo_totalbet: {self.result['oppo_totalbet'][i]}, "
                f"Aggression: {self.result['aggression'][i]}, "
                f"oppo_fold: {self.result['oppo_fold'][i]}"

            )
            # print(type(desc))
            prompts.append(desc)
        df = pd.DataFrame({"prompts":prompts, "value":self.value_list})
        if os.path.exists(self.path):
            df1 = pd.read_csv(self.path)
            df0 = pd.concat([df1, df])
            df0.to_csv("data.csv")
        else:
            df.to_csv("data.csv")

    def report_prompts(self):
        prompts = []
        for i in range(len(self.result["stage"])):
            # 格式化历史动作
            formatted_actions = " | ".join(
                [",".join(round_actions) for round_actions in self.result["history_action"][i]]
            )
            # print(self.result["bet_history"])
            # 格式化押注记录
            formatted_bets = " | ".join(
                [f"Round{idx+1}:"+",".join(map(str,bets)) 
                 for idx, bets in enumerate(self.result["bet_history"][i])]
            )
            # 构建描述字符串
            desc = (
                f"Stage: {self.result['stage'][i].lower()}, "
                f"Private Cards: {', '.join(str(card) for card in self.result['private_cards'][i]) if self.result['private_cards'][i] else 'None'}, "
                f"Public Cards: {', '.join(str(card) for card in self.result['public_cards'][i]) if self.result['public_cards'][i] else 'None'}, "
                f"Action: {self.result['action'][i]}, "
                f"History Actions: [{formatted_actions}], "
                f"Bet History: [{formatted_bets}], "
                f"Chips: {self.result['hand_chips'][i]}, "
                f"Totalbet: {self.result['totalbet'][i]}, "
                f"Player: {self.result['player'][i]}, "
                f"Location: {self.result['location'][i]}, "
                f"Oppo_totalbet: {self.result['oppo_totalbet'][i]}, "
                f"Aggression: {self.result['aggression'][i]}, "
                f"oppo_fold: {self.result['oppo_fold'][i]}"

            )
            # print(type(desc))
            prompts.append(desc)
        return prompts

# 判断是否是顺子
def is_straight(ranks):
    if ranks == [14, 5, 4, 3, 2]:
        return True
    return sorted(ranks) == list(range(min(ranks), min(ranks) + 5))

# 判断是否是同花
def is_flush(suits):
    return len(set(suits)) == 1

# 判断牌型
def evaluate_hand(hand):
    ranks = [RANKS[card[0]] for card in hand]
    suits = [card[1] for card in hand]
    rank_counts = Counter(ranks)

    # 皇家同花顺
    if is_flush(suits) and is_straight(ranks) and max(ranks) == 14:
        return "Royal Flush", sorted(ranks, reverse=True)
    # 同花顺
    elif is_flush(suits) and is_straight(ranks):
        return "Straight Flush", sorted(ranks, reverse=True)
    # 四条
    elif any(count == 4 for count in rank_counts.values()):
        quad_rank = max((rank for rank, count in rank_counts.items() if count == 4), default=None)
        kicker = max((rank for rank, count in rank_counts.items() if count != 4), default=None)
        return "Four of a Kind", [quad_rank] * 4 + [kicker]
    # 葫芦
    elif any(count == 3 for count in rank_counts.values()) and any(count == 2 for count in rank_counts.values()):
        triple_rank = max((rank for rank, count in rank_counts.items() if count == 3), default=None)
        pair_rank = max((rank for rank, count in rank_counts.items() if count == 2), default=None)
        return "Full House", [triple_rank] * 3 + [pair_rank] * 2
    # 同花
    elif is_flush(suits):
        return "Flush", sorted(ranks, reverse=True)
    # 顺子
    elif is_straight(ranks):
        return "Straight", sorted(ranks, reverse=True)
    # 三条
    elif any(count == 3 for count in rank_counts.values()):
        triple_rank = max((rank for rank, count in rank_counts.items() if count == 3), default=None)
        kickers = sorted((rank for rank, count in rank_counts.items() if count != 3), reverse=True)
        return "Three of a Kind", [triple_rank] * 3 + kickers[:2]
    # 两对
    elif len([rank for rank, count in rank_counts.items() if count == 2]) == 2:
        pairs = sorted((rank for rank, count in rank_counts.items() if count == 2), reverse=True)
        kicker = max((rank for rank, count in rank_counts.items() if count != 2), default=None)
        return "Two Pair", pairs * 2 + [kicker]
    # 一对
    elif any(count == 2 for count in rank_counts.values()):
        pair_rank = max((rank for rank, count in rank_counts.items() if count == 2), default=None)
        kickers = sorted((rank for rank, count in rank_counts.items() if count != 2), reverse=True)
        return "One Pair", [pair_rank] * 2 + kickers[:3]
    # 高牌
    else:
        return "High Card", sorted(ranks, reverse=True)

# 主函数
def best_hand(public_cards, hole_cards):
    while len(public_cards) < 5:
        public_cards.append('Bk')
    all_cards = public_cards + hole_cards
    best_hand_type = None
    best_hand_value = None

    # 从 7 张牌中选择 5 张牌的所有组合
    from itertools import combinations
    for combo in combinations(all_cards, 5):
        hand_type, hand_value = evaluate_hand(combo)
        if best_hand_type is None or compare_hands(hand_type, hand_value, best_hand_type, best_hand_value) > 0:
            best_hand_type, best_hand_value = hand_type, hand_value

    return best_hand_type, best_hand_value

# 比较两个手牌的大小
def compare_hands(type1, value1, type2, value2):
    order = ["High Card", "One Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"]
    if order.index(type1) > order.index(type2):
        return 1
    elif order.index(type1) < order.index(type2):
        return -1
    else:
        for v1, v2 in zip(value1, value2):
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0
    

def transfer_cards(table_cards):
        cards = []
        for i in table_cards:
            inner_sequennce = i % 13
            if i//13 == 0:
                card_str = card_sequence[inner_sequennce] + "s"
                cards.append(card_str)
            elif i//13 == 1:
                card_str = card_sequence[inner_sequennce] + "d"
                cards.append(card_str)
            elif i//13 == 2:
                card_str = card_sequence[inner_sequennce] + "h"
                cards.append(card_str)
            elif i//13 == 3:
                card_str = card_sequence[inner_sequennce] + "c"
                cards.append(card_str)            
        return cards
   
# 计算手牌和公共牌的最大匹配，折算惩罚率
def calculate_max_match(hand_cards_raw:list, public_cards_raw:list)->float:
    hand_cards = transfer_cards(hand_cards_raw)
    # public_cards = transfer_cards(public_cards_raw)
    # hand_cards = hand_cards_raw
    public_cards = public_cards_raw
    card_type = None

    if hand_cards == []:
        hand_res = 1
        public_res = 1
    
    else:
        if hand_cards[0][0] == hand_cards[1][0]:
            card_type = "pair"
        elif hand_cards[0][1] == hand_cards[1][1]:
            card_type = "flush"
        else:
            card_type = "mixed"
    
        index_key = (hand_cards[0][0]+" "+hand_cards[1][0], card_type)
        if index_key not in win_rate.keys():
            index_key = (hand_cards[1][0]+" "+hand_cards[0][0], card_type)
        hand_res = win_rate[index_key]
        
        # best_match = "类型：如同花顺等"
        best_match, _ = best_hand(public_cards.copy(), hand_cards.copy())
        public_res = win_rate_heads_up[best_match]
        # print(hand_res, public_res)
    return 0.5*hand_res+0.5*public_res
    
    
def arg_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument("--path", type=str, default="log", help="the path of log")
    dir_lis = os.listdir("log")
    file_lis = os.listdir("log/"+dir_lis[1])
    parse.add_argument("--name", type=str, default=dir_lis[0]+"/"+file_lis[0], help="the name of log_file")
    
    parse.add_argument("--nick", type=str, default="p_13304936695", help="the player you want to analyze")
    parse.add_argument("--opponent", type=str, default="p_13304936695_player1", help="the opponent")
    
    args = parse.parse_args()
    return args


# 修改round检测逻辑为判断牌桌上公开牌的数量
def extract_player_info(data, player):
    batches = []
    for game in data:
        this_game_batches = []
        basic_info = game.get("basic_info", {})
        # print("basic", basic_info)
        dynamic_info = game.get("dynamic_info", {})
        # print("dynamic", dynamic_info)
        round_history = dynamic_info.get("round_history", [])
        # print("round", round_history)

        # 初始化玩家和对手的信息
        player_info = None
        init_hand_chips = 2000

        # 提取玩家和对手的基本信息
        for seat in basic_info.get("seat_info", []):
            # print(seat['seatid'])
            if seat.get("usrname") == player:
                # print("find player")
                player_info = seat["seatid"]
                init_hand_chips = seat["hand_chips"]
                break
        # 需要拿到一局的信息，而不是多局拼在一起，如何解析是一个问题
        player_chips_now = init_hand_chips
        processor = RoundInfoManager(init_hand_chips)
        batch = {}
        p_flag = False
        win_flag = True
        totalbet = 0
        oppo = {}
        hand_cards = []

        for round in round_history:
            # print(round)
            cards_num = 5-round["table_cards"].count(-1)

            if cards_num == 0:
                if this_game_batches != []:
                    for batch in reversed(this_game_batches):
                        if "hand_cards" not in batch:
                            this_game_batches.pop()
                        else:
                            break
                    batches.append(this_game_batches)
                    this_game_batches = []
            totalbet += round["bet"]
            # print(cards_num)
            # print(player_info, round["player"])
            # print("="*20)

            if round["player"] != player_info:
               
                act = ""
                if round["type"] == 2:
                    act = "bet"
                if round["type"] == 5:
                    act = "fold"

                if round["player"] not in oppo:
                    new_oppo = OpponentAnalysis(round["player"])
                    oppo[round["player"]] = new_oppo

                oppo[round["player"]].calculate_new_aggression(round["type"])

                p_flag, isdone = processor.get_extern_info(cards_num, act, round["bet"])
                # print(p_flag, isdone)
                if p_flag:
                    p_flag = False
                    
                    if win_flag:
                        batch["winner"] = 1
                    else:
                        batch["winner"] = -1
                        
                    if basic_info['blind']['big_blind']['seatid'] == round["player"]:
                        batch["location"] = 1
                    elif basic_info['blind']['small_blind']['seatid'] == round["player"]:
                        batch["location"] = 2
                    elif basic_info['dealer_info']['seatid'] == round["player"]:
                        batch["location"] = 3
                    else:
                        batch["location"] = 4
                        
                    batch["history_action"], batch["bet_history"], batch["stage"] = processor.report_history()
                    batch["totalbet"] = totalbet
                    # print(totalbet)
                    batch["player"] = round["player"]
                    oppo_repo = oppo[round["player"]].report()
                    batch["oppo_totalbet"] = oppo_repo["total_bet"]
                    batch["aggression"] = oppo_repo["aggression"]
                    batch["oppo_fold"] = oppo_repo["fold"]

                    this_game_batches.append(batch)
                    # if "hand_cards" not in batch.keys():
                        # print("no hand_cards")
                    batch = {}
                    
                    if isdone:
                        processor.update_round(round["player_chips"])
                        # print(this_game_batches)
                        batches.append(this_game_batches)
                        # print(this_game_batches)
                        this_game_batches = []
                        totalbet = 0
                    
            else:
                # print("="*10)
                act = ""
                if round["type"] == 2:
                    act = "bet"
                if round["type"] == 5:
                    act = "fold"

                batch["hand_chips"] = round["player_chips"]
                if -1 in round["table_cards"]:
                    table_valid = round["table_cards"].remove(-1)
                else:
                    table_valid = round["table_cards"]
                batch["public_cards"] = table_valid
                batch["action"] = act
                batch["hand_cards"] = round["hand_cards"]
                # print(batch)
                # 判断一局的胜负
                if processor.stage == "RIVER":
                    if round["player_chips"] < player_chips_now:
                        win_flag = False
                    else:
                        win_flag = True
    return batches
 
 
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
    
def main():
    # args = arg_parse()
    # log_file = args.path + "/" + args.name
    # print(args.nick)

    log_folder = "log"
    data = []
    for subdir in os.listdir(log_folder):
        subdir_path = os.path.join(log_folder, subdir)

        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".json"):
                    log_file = os.path.join(subdir_path, file)
                    print(log_file)
                    data.extend(load_json_file(log_file))
                    print(len(data))

    for subdir in os.listdir(log_folder):
        subdir_path = os.path.join(log_folder, subdir)
        
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".json"):
                    log_file = os.path.join(subdir_path, file)
                    print(log_file)
                    
                    try:
                        # 尝试解析JSON文件
                        with open(log_file, 'r') as f:
                            json.load(f)  # 仅检查格式，不保存数据
                    except json.JSONDecodeError as e:
                        print(f"JSON格式错误在文件 {log_file}: {e}")
                        continue  # 跳过后续处理
                    except Exception as e:
                        print(f"读取文件时发生错误 {log_file}: {e}")
                        continue
                    data.extend(load_json_file(log_file))

    player = "p_13304936695"
    batches = extract_player_info(data, player)
    print("batches", len(batches))
    scorer = LabelScorer()
    for i in batches:
        game1 = node_list()

        if i == []:
            continue
        for k in i:
            print(k)
            node = Node()
            game1.add_node(node, k)
        
        c_puct = calculate_max_match(k["hand_cards"], k["public_cards"])
        game1.backbone(c_puct=c_puct)
        scorer.get_result(game1)
    scorer.report_all_result()

if __name__ == "__main__":
    main()