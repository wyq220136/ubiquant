import json
import os
from datetime import datetime
import pandas as pd
from opposetbuild import PokerStateEncoder

card_sequence = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

class player_info:
    def __init__(self, seatid, name:str, bet):
        self.seatid = seatid
        self.usrname = name
        self.action_sequence = []
        self.hand_chips = bet
        self.is_active = True
    
    
    def update_action(self, action):
        act_tmp = action_info(action["Type"], action["Bet"])
        self.action_sequence.append(act_tmp)
    
    def update_handchips(self, bet):
        self.hand_chips = bet
    
    def report(self) -> dict:
        return{
            "seatid":self.seatid, 
            "usrname":self.usrname,
            "hand_chips":self.hand_chips
        }

        
class action_info:
    """
    记录玩家动作
    type:动作类型
    bet:下注金额
    """
    def __init__(self, type, bet):
        self.type = type
        self.bet = bet

    def report(self) -> dict:
        return {
            "type":self.type,
            "bet":self.bet
        }
        
class round_info(action_info):
    def __init__(self, type, bet, player_id, round_num, cards):
        super().__init__(type, bet)
        self.player = player_id
        self.round_num = round_num
        self.table_cards = cards
        self.stage = stage()
    
    def load_player_cards(self, cards:list):
        self.player_cards = cards
        
    
    def load_hand_chips(self, chips:int):
        self.player_chips = chips
        
    def report(self) -> dict:
        return {
            "player":self.player,
            "type":self.type,
            "bet":self.bet,
            # "round_num":self.round_num,
            "table_cards":self.table_cards,
            "stage":self.stage.report(self.round_num),
            "player_chips":self.player_chips,
            "hand_cards":self.player_cards
        }

class stage:
    def __init__(self):
        self.stage = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
        
    def report(self, idx:int):
        if idx == 101:
            return "SHOWDOWN"
        return self.stage[idx]


class static_info:
    def __init__(self):
        # 大盲注和小盲注
        self.blind = {}

    def load_static_info(self, small_blind: int, big_blind: int, dealer: int, player_list, bet_list:list):
        self.blind["big_blind"] = player_info(big_blind, player_list[big_blind], bet_list[big_blind])
        self.blind["small_blind"] = player_info(small_blind, player_list[small_blind], bet_list[small_blind])
        
        self.dealer_info = player_info(dealer, player_list[dealer], bet_list[dealer])
        self.bet_list = bet_list
        # dict
        self.seat_info = self.valid_player(player_list)
        
    # 识别有效玩家   
    def valid_player(self, p_list):
        p_valid = {}
        for p in range(len(p_list)):
            if p_list[p] != '':
                player = player_info(p, p_list[p], self.bet_list[p])
                p_valid[p] = player
        return p_valid
        
    def update_player_info(self, batch, bet):
        id = batch["SeatId"]
        self.seat_info[id].update_action(batch)
        for idx in self.seat_info.keys():
            self.seat_info[idx].update_handchips(bet[idx])
            
    def report_player_info(self):
        info = []
        for i in self.seat_info.values():
            a = i.report()
            info.append(a)
        return info
    
    def report(self)->dict:
        return {
            "blind": {
                "big_blind": self.blind["big_blind"].report(),
                "small_blind": self.blind["small_blind"].report()
            },
            "seat_info": self.report_player_info(),
            "dealer_info": self.dealer_info.report()  # 修正拼写
        }


class dynamic_info:
    def __init__(self):
        self.round_info = []
        self.stage_now = 0
        
    def copy_nick_info(self, player_info:dict):
        self.nick_info = player_info
        
        
        
    def update_round_info(self, action, round_num, table_cards, hand_cards:list):
        self.cards_string = self.transfer_cards(table_cards)
        tmp = round_info(action["Type"], action["Bet"], action["SeatId"], round_num, self.cards_string)
        tmp.load_hand_chips(self.nick_info[action["SeatId"]].hand_chips)
        tmp.load_player_cards(hand_cards)
        self.round_info.append(tmp)
        self.stage_now = round_num
        self.tablecards = table_cards
        
        
    def transfer_cards(self, table_cards):
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

                
    def report(self) -> dict:
        return {
            "round_history":[i.report() for i in self.round_info]
        }
        

# 可能存在多个room同时进行游戏，将total_info作为第二级信息整合源
class total_info:
    def __init__(self, rid:str):
        self.basic_info = static_info()
        self.dynamic_info = dynamic_info()
        self.logger = logger(rid)  # 新日志实例
        self.winner = None
        self.isdone = False # 记录对局是否完成
        self.player_cards = {}
        
        self.hero_name = "p_13304936695"
        # self.lock = threading.Lock()
        
        self.bet_all = 0
        
        # 编码状态信息
        self.state_encoder = PokerStateEncoder()
        
    
    def load_static(self, small_blind: int, big_blind: int, dealer: int, player_list: list, bet_list:list):
        self._init_params = (small_blind, big_blind, dealer, player_list.copy(), bet_list.copy())
        sb, bb, dl, pl, b1 = self._init_params
        self.basic_info.load_static_info(sb, bb, dl, pl, b1)
        self.dynamic_info.copy_nick_info(self.basic_info.seat_info)
        for i in self.basic_info.seat_info.values():
            if i.usrname == self.hero_name:
                self.state_encoder.player_seat_id = i.seatid
        data = self.basic_info.report()
        self.state_encoder.load_all_nick(data)
        
        
    def report(self)->dict:
        dic = {
            "basic_info":self.basic_info.report(),
            "dynamic_info":self.dynamic_info.report(),
            "winner":self.winner
        }
        return dic
    
    def keep(self):
        if self.isdone:
            self.logger.write_log(self.report())
            self.isdone = False
        
    def update_all(self, batch, round_num, bet, table_cards, pot_bet):
        self.basic_info.update_player_info(batch, bet)
        self.dynamic_info.update_round_info(batch, round_num, table_cards, self.player_cards.get(batch["SeatId"], []))
        
        # 计算当前池底的大小
        self.bet_all = sum(pot_bet)
        # self.report_rl()
        if batch["SeatId"] != self.state_encoder.player_seat_id:
            self.state_encoder.update_chips(self.dynamic_info.nick_info[batch["SeatId"]].hand_chips, batch["SeatId"])
            
            act = None
            if batch["Type"] == 5:
                act = 0
            else:
                act = 1
            
            self.state_encoder.update_oppoaction(act, batch["SeatId"])
        
        if batch["Type"] == 5:
            self.basic_info.seat_info[batch["SeatId"]].is_active = False
            
        if round_num == 4:
            self.isdone = True
        
    
    def set_winner(self, winner:dict):
        self.winner = list(winner.keys())[0]
       
        
    def read_cards(self):
        return self.dynamic_info.cards_string
    
    def update_player_cards(self, nick:int, cards:list):
        self.player_cards[nick] = cards

        
    def get_player_cards(self, nick:str)->list:
        if nick not in self.player_cards: 
            return []
        return self.player_cards[nick]
    
    # 为了编码单独封出来功能接口
    def report_rl(self, action, round_num, table_cards):
        tmp_encode = round_info(action["Type"], action["Bet"], action["SeatId"], round_num, table_cards)
        tmp_encode.load_hand_chips(self.dynamic_info.nick_info[action["SeatId"]].hand_chips)
        round_res = tmp_encode.report()
        round_res["all_bet"] = self.bet_all
        # del tmp_encode
        state_vector = self.state_encoder.encode(round_res)
        return state_vector
    
    
class logger:
    def __init__(self, path):
        log_dir = "log/"+path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(log_dir + f"/log{timestamp}.json", "w")

    def write_log(self, dict):
        js = json.dumps(dict, sort_keys=True, indent=4, separators=(',',':'))
        self.log_file.write(js)
        self.log_file.write('\n\n')

    
# 整合所有room的信息，total_info的父级
class all_room_info:
    def __init__(self):
        # 这一级维护的信息只有房间号和房间数
        self.room_manage = {}
        self.room_number = 0
        self.linear_data = None
        
    def load_room(self, rid:str):
        if rid not in self.room_manage.keys():
            self.room_manage[rid] = total_info(rid)
            self.room_number += 1
        
            
    def update_room(self, rid:str, batch, round_num, bet, table_cards, pot_bet):
        self.room_manage[rid].update_all(batch, round_num, bet, table_cards, pot_bet)
        
        if self.linear_data == None:
            tmp_list = []
            for i in self.room_manage[rid].basic_info.seat_info.values():
                nick = i.usrname
                pos = i.seatid
                if pos == self.room_manage[rid].basic_info.dealer_info.seatid:
                    pos = "dealer"
                elif pos == self.room_manage[rid].basic_info.blind["small_blind"].seatid:
                    pos = "small_blind"
                elif pos == self.room_manage[rid].basic_info.blind["big_blind"].seatid:
                    pos = "big_blind"
                else:
                    pos = "normal"
                tmp_list.append((nick, pos))
            self.linear_data = linearRegAnalysis(tmp_list, rid)
        else:
            if rid == self.linear_data.room_id:
                nick_name = self.room_manage[rid].basic_info.seat_info[batch["SeatId"]].usrname
                self.linear_data.update_info(self.room_manage[rid].dynamic_info.stage_now, nick_name, batch["Bet"], self.room_manage[rid].get_player_cards(nick_name))
        
    def load_room_static(self, rid:str, small_blind: int, big_blind: int, dealer: int, player_list: list, bet_list:list):
        self.room_manage[rid].load_static(small_blind, big_blind, dealer, player_list, bet_list)

    def keep_all(self):
        for room in self.room_manage.values():
            room.keep()
        if self.linear_data != None:
            self.linear_data.output_all()
            
    def set_room_winnner(self, rid:str, winner):
        self.room_manage[rid].set_winner(winner)
        
    def update_nick_cards(self, rid:str, nick:int, cards:list):
        self.room_manage[rid].update_player_cards(nick, cards)
 
# 需要一个面向prompts的接口
 
class playerAnalysis:
    def __init__(self, name:str, seat:int):
        self.name = name
        self.bet = []
        # 庄家是0，大盲是2.小盲是1
        self.pos = 0 if seat == "dealer" else (2 if seat == "big_blind" else (1 if  seat == "small_blind" else -1))
        self.average_bet = []
        self.cards = []
        self.hand_cards = ""
    
    def getpoker(self, cards:list):
        init_str = ""
        for card in cards:
            init_str += card + " "
        init_str.strip(" ")
        self.cards.append(init_str)
        # 手牌信息原来和下注信息同步更新，但是如果没有下注就用现在的hand_cards更新
        self.hand_cards = cards
        
    def new_action(self, bet, cards:list):
        self.getpoker(cards)
        self.bet.append(bet)
        self.average_bet.append(sum(self.bet)/len(self.bet) if len(self.bet) > 0 else 0)
        
    def output(self, round_list:list, stage_list:list):
        pos_list = [self.pos]*len(round_list)
        # print("wyq220136", len(round_list), len(stage_list), len(self.bet))
        # 2000个数据记录一次
        if len(round_list) >= 2000:
            df_tmp = pd.DataFrame({"round":round_list, "stage":stage_list, "bet":self.bet, "average_bet":self.average_bet, "cards":self.cards, "pos":pos_list})
            df_tmp.to_csv(f"{self.name}_{self.pos}", index=False)
        
  
class linearRegAnalysis:
    def __init__(self, valid_player:list, id:str):
        path = "linearReg"
        if not os.path.exists(path):
            os.makedirs(path)
        self.round_num = []
        self.round_conter = 1
        self.stage_num = []
        self.last_stage = 0
        # 取指定房间的数据
        self.room_id = id
        
        # 数据格式valid_player = [(name, pos)...]
        self.player_store = {key:playerAnalysis(key, i) for key, i in valid_player}
        
    # 直接在all_info中获得所有信息
    def update_info(self, stage:int, name:str, bet:int, cards:list):
        self.stage_num.append(stage)
        if stage == 0 and (self.last_stage == 4 or self.last_stage == 101):
            self.round_conter += 1
        self.round_num.append(self.round_conter)
        self.last_stage = stage
        # print(list(self.player_store.keys()))
        self.player_store[name].new_action(bet, cards)
        for key in self.player_store:
            if key != name:
                self.player_store[key].new_action(0, self.player_store[key].hand_cards)
         
    def output_all(self):
        for key in self.player_store:
            self.player_store[key].output(self.round_num, self.stage_num)
            