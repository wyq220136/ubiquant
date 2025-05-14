'''
    AI: v1_1版本
    详见AI-v1.1_interpretation.txt
'''
import random
import time


def ai(table_info,hand_cards):
    # 
    BetLimit = table_info['GameStatus']['NowAction']['BetLimit']
    id=table_info['GameStatus']['NowAction']['SeatId']
    # 牌型权重系数（实际上是10个等级，0只是占位）
    # 牌面等级：高牌 1	一对 2	两对 3	三条 4	顺子 5	同花 6	葫芦 7	四条 8	同花顺 9	皇家同花顺：10
    # 牌面权值：1	2	4	8	15	32	64	128	256	512
    weight = [0, 1, 2, 4, 8, 16, 32, 64, 64, 64, 64]
    # 修改权重的原因是-如果极端牌型的权重过高，程序会过于乐观；同时在葫芦及以上，都能够在大多数情况下确保赢
    # 初始时没有发牌，剩余的牌是全的
    remain_card = list(range(0, 52))
    # table_info['TableStatus']['TableCard']是每一个玩家的手牌，hand_cards是公开的牌？计算已发出的牌
    cards = [x for x in table_info['TableStatus']['TableCard'] if x != -1] + hand_cards
    num = len(cards)
    # 把所有已发出的牌从剩余牌中拿出去
    for x in cards:
        remain_card.pop(remain_card.index(x))
    
    # 这个数组用于存储上述牌型分别出现了几次
    cnt = [0 for col in range(11)]
    # 模拟发牌1000次
    for i in range(1000): # 1000次需要根据运行时间进行调整
        heap = remain_card[:]
        mycards = cards[:]
        random.shuffle(heap)
        # 补牌至7张
        while len(mycards) != 7:
            mycards.append(heap.pop())
        hand = Hand(mycards)
        level = hand.level
        cnt[level] += weight[level]

    # sum为评估值
    sum = 0
    for x in cnt:
        sum += x / 1000

    decision = {}
    decision["callbet"]=0
    totalbet = 0

    # limit是一个外部输入数组，具体内容不详
    delta=BetLimit[0] # 跟注金额
    minbet=max( BetLimit[1] ,0)
    small_b = table_info['GameStatus']['SBCur']
    big_b = table_info['GameStatus']['BBCur']
    dealer = table_info['GameStatus']['DealerCur']
    handchips = table_info['TableStatus']['User']['HandChips'][id]
    # delta = state.minbet - state.player[state.currpos].bet
    # if delta >= state.player[state.currpos].money:
    #     totalbet = 1000
    # else:
    #     totalbet = state.player[state.currpos].totalbet + state.minbet
    totalbet= table_info['TableStatus']['User']['TotalBet'][id]+minbet
    if num == 2:

        point0, point1 = cards[0] // 4, cards[1] // 4
        max_point = max(point0, point1)
        min_point = min(point0, point1)
        is_pair = (point0 == point1)
        is_suited = (cards[0] % 4) == (cards[1] % 4)
        gap = abs(point0 - point1)
        
        if is_pair:
            # 对子逻辑
            if max_point >= 12:  # 对A（假设点数12为A）
                if totalbet < 300:
                    decision = add_bet(table_info, 300)
                else:
                    decision["callbet"] = 1
            elif max_point >= 10:  # 对K、对Q
                if totalbet < 200:
                    decision = add_bet(table_info, 200)
                else:
                    decision["callbet"] = 1
            else:  # 中小对子
                if totalbet <= 100:
                    decision["callbet"] = 1
                else:
                    decision["giveup"] = 1
        else:
            # 非对子逻辑
            is_high_card = (max_point >= 10)  # J及以上
            is_ace_high = (max_point == 12) and (min_point >= 8)  # A带T+
            if is_suited and is_high_card:
                # 同花高牌激进处理
                if totalbet <= 200:
                    decision["callbet"] = 1
                else:
                    decision["giveup"] = 1
            elif is_ace_high or (gap == 1 and max_point >= 9):
                # 连张或A带高牌
                if totalbet <= 150:
                    decision["callbet"] = 1
                else:
                    decision["giveup"] = 1
            else:
                # 其他情况弃牌
                decision["giveup"] = 1

    elif num == 5:
        current_bet = BetLimit[1]
        decision = {"callbet": 0, "raisebet": 0, "giveup": 0, "allin": 0, "check": 0}
        # 基础筹码检查函数
        def check_chips(required):
            if required > handchips:
                decision["allin"] = 1
                return handchips
            return required

        if sum < 4:
            # 极低牌力时优先利用check选项
            if delta == 0 and BetLimit[0] == 0:
                decision["check"] = 1
            else:
                decision["giveup"] = 1

        elif 4 <= sum < 10:
            # 复合决策条件：筹码深度+位置优势
            blind_condition = (id == small_b or id == big_b) and delta <= big_b * 2
            if totalbet < 150 or delta <= 50 or blind_condition:
                if delta > handchips:
                    decision["allin"] = 1
                else:
                    decision["callbet"] = 1
                    # 特殊场景：当处于有利位置且筹码较深时小额加注
                    if id == dealer and handchips > 1000 and delta == 0:
                        decision["raisebet"] = 1
                        decision["callbet"] = 0
                        decision["amount"] = check_chips(100)
            else:
                decision["giveup"] = 1

        elif 10 <= sum < 20:
            # 中等牌力采用动态加注策略
            if totalbet < 200:
                raise_amount = max(2.5*current_bet, 205)  # 动态计算加注额度
                actual_raise = check_chips(raise_amount)
                if actual_raise > current_bet:
                    decision["raisebet"] = 1
                    decision["amount"] = actual_raise
                    if actual_raise == handchips:
                        decision["allin"] = 1
            else:
                if delta > handchips:
                    decision["allin"] = 1
                else:
                    decision["callbet"] = 1

        elif 20 <= sum < 50:
            # 强牌采用激进加注策略
            if totalbet < 300:
                raise_amount = max(3*current_bet, 305)
                actual_raise = check_chips(raise_amount)
                decision["raisebet"] = 1
                decision["amount"] = actual_raise
                if actual_raise == handchips:
                    decision["allin"] = 1
            else:
                if delta > handchips:
                    decision["allin"] = 1
                else:
                    decision["callbet"] = 1

        else:  # sum >=50
            # 超强牌直接全押或最大化加注
            if delta > 0:
                decision["raisebet"] = 1
                decision["amount"] = check_chips(handchips)
                if decision["amount"] == handchips:
                    decision["allin"] = 1
            else:
                decision["raisebet"] = 1
                decision["amount"] = check_chips(max(current_bet*4, 500))

        # Check逻辑优化（仅在无代价时且未做其他决策时触发）
        if BetLimit[0] == 0 and delta == 0:
            if not any([decision["callbet"], decision["raisebet"], decision["giveup"], decision["allin"]]):
                decision["check"] = 1
        
        # 冲突解决优先级：allin > raisebet > callbet > check > giveup
        if decision["allin"]:
            decision.update({"callbet":0, "raisebet":0, "check":0, "giveup":0})
        elif decision["raisebet"]:
            decision.update({"callbet":0, "check":0, "giveup":0})
        elif decision["callbet"]:
            decision.update({"check":0, "giveup":0})
        elif decision["check"]:
            decision["giveup"] = 0
        
        # 强制弃牌保护（当需要跟注金额超过心理阈值时）
        if delta > 0 and not any([decision["callbet"], decision["raisebet"], decision["allin"], decision["check"]]):
            decision["giveup"] = 1

    elif num == 6:
        decision = {"giveup": 0, "callbet": 0, "raisebet": 0, "allin": 0, "check": 0}

        # 基础决策逻辑
        if sum < 2:
            decision["giveup"] = 1
        elif 2 <= sum < 8:
            if totalbet <= 200:
                decision["callbet"] = 1
            else:
                decision["callbet"] = 1 if delta <= min(40, handchips) else 0
                decision["giveup"] = 1 if delta > min(40, handchips) else 0
        elif 8 <= sum < 20:
            if totalbet < 300:
                needed = 305 - totalbet
                if needed <= handchips:
                    decision["raisebet"] = 1
                    decision["amount"] = 305
                else:
                    decision["allin"] = 1
            else:
                if delta <= handchips:
                    decision["callbet"] = 1
                else:
                    decision["allin"] = 1
        elif 20 <= sum < 40:
            if totalbet < 400:
                target = 605 if id == dealer else 405
                needed = target - totalbet
                if needed <= handchips:
                    decision["raisebet"] = 1
                    decision["amount"] = target
                else:
                    decision["allin"] = 1
            else:
                if delta <= handchips:
                    decision["callbet"] = 1
                else:
                    decision["allin"] = 1
        else:
            decision["allin"] = 1

        # 强制覆盖逻辑：当允许check时（无需跟注）
        if BetLimit[0] == 0 and delta == 0:
            decision = {"giveup": 0, "callbet": 0, "raisebet": 0, "allin": 0, "check": 1}
        
        # 互斥逻辑处理
        if decision["allin"]:
            decision["giveup"] = 0
            decision["callbet"] = 0
            decision["raisebet"] = 0
            decision["check"] = 0

        # 筹码不足保护
        if decision.get("raisebet") and decision["amount"] > totalbet + handchips:
            decision["raisebet"] = 0
            decision["allin"] = 1

        # 跟注保护
        if decision["callbet"] and delta > handchips:
            decision["callbet"] = 0
            decision["allin"] = 1

    elif num == 7:
        decision = {'allin': 0, 'callbet': 0, 'raisebet': 0, 'giveup': 0, 'check': 0, 'amount': 0}
        
        if level == 7:
            decision['allin'] = 1
            
        elif level == 6:
            # 跟注，若加注后筹码不足则全下
            if totalbet < 400:
                # 庄家加注到605，其他玩家到405
                raise_amount = 605 if id == dealer else 405
                decision = add_bet(table_info, raise_amount)
                if decision['amount'] > handchips:
                    decision['allin'] = 1
                    decision['raisebet'] = 0
                    decision['amount'] = handchips
            else:
                # 跟注时检查筹码是否足够
                if delta > handchips:
                    decision['allin'] = 1
                    decision['callbet'] = 0
                else:
                    decision['callbet'] = 1

        elif level == 5:
            # 加注到405或全下
            if totalbet < 400:
                decision = add_bet(table_info, 405)
                if decision['amount'] > handchips:
                    decision['allin'] = 1
                    decision['amount'] = handchips
            else:
                if delta > handchips:
                    decision['allin'] = 1
                else:
                    decision['callbet'] = 1

        elif level == 4:
            # 加注到405时处理筹码不足
            if totalbet < 400:
                decision = add_bet(table_info, 405)
                if decision['amount'] > handchips:
                    decision['allin'] = 1
                    decision['amount'] = handchips
            else:
                decision['callbet'] = 1

        elif level == 3:
            if totalbet > 500:
                decision['giveup'] = 1
            elif totalbet < 300:
                decision = add_bet(table_info, 300)
                if decision['amount'] > handchips:
                    decision['allin'] = 1
                    decision['amount'] = handchips
            else:
                if delta > handchips:
                    decision['allin'] = 1
                else:
                    decision['callbet'] = 1

        elif level == 2:
            # 修正手牌判断逻辑（假设前两张是手牌）
            hand = cards[:2]
            aces = hand.count(0) >= 2
            kings = hand.count(12) >= 2
            
            if aces or kings:
                if totalbet <= 200:
                    decision['callbet'] = 1
                elif delta <= 50:
                    decision['callbet'] = 1
                else:
                    decision['giveup'] = 1
            else:
                if totalbet <= 200:
                    decision['callbet'] = 1
                else:
                    decision['giveup'] = 1

        elif level == 1:
            decision['giveup'] = 1

        else:
            print(f'Invalid level {level} with {num} cards')
            assert 0

        # BetLimit处理（需要确保不覆盖allin状态）
        if decision['callbet'] and BetLimit[0] == 0 and decision['allin'] == 0:
            if random.randint(0, 2) == 0:
                min_raise = max(table_info['BB'], delta*2)
                decision.update({
                    'callbet': 0,
                    'raisebet': 1,
                    'amount': min_raise
                })

    return decision


# add_bet: 将本局总注额加到total

def add_bet(table_info, total):
    # amount: 本局需要下的总注
    BetLimit = table_info['GameStatus']['NowAction']['BetLimit']
    seatId=table_info['GameStatus']['NowAction']['SeatId']
    amount = total - table_info['TableStatus']['User']['TotalBet'][seatId]
    assert(amount > table_info['TableStatus']['User']['TotalBet'][seatId])
    # Obey the rule of last_raised
    decision = {'callbet':0}
    if amount >=BetLimit[1] and amount <=BetLimit[2]:
        decision["raisebet"] = 1
        decision["amount"] = amount
    else:
        if BetLimit[1]!=-1:
            decision["raisebet"] = 1
            decision["amount"] = amount
        else:
            decision["callbet"] = 1
    return decision

from time import sleep

# V1.4
# 0 黑桃 1 红桃 2 方片 3 草花
# 牌的id: 0-51

'''
牌面level编号
    皇家同花顺：10
    同花顺    ：9
    四条      ：8
    葫芦      ：7
    同花      ：6
    顺子      ：5
    三条      ：4
    两对      ：3
    一对      ：2
    高牌      ：1
'''

'''
DealerRequest message Definition:
type:
    0   heartbeat
    1   response from server for state update
    2   request from server for decision
    3   request from server for state control
    4   response from server for client init
    5   response from server for game over
status:
    -1  uninitialized
'''

# alter the card id into color
def id2color(card):
    return card % 4

# alter the card id into number
def id2num(card):
    return card // 4

'''
hand.level
牌面等级：高牌 1	一对 2	两对 3	三条 4	顺子 5	同花 6	葫芦 7	四条 8	同花顺 9	皇家同花顺：10

'''
def judge_exist(x):
    if x >= 1:
        return True
    return False

# poker hand of 7 card
class Hand(object):
    def __init__(self, cards):
        cards = cards[:]
        self.level = 0
        self.cnt_num = [0] * 13
        self.cnt_color = [0] * 4
        self.cnt_num_eachcolor = [[0 for col in range(13)] for row in range(4)]
        self.maxnum = -1
        self.single = []
        self.pair = []
        self.tripple = []
        self.nums = []
        for x in cards:
            self.cnt_num[id2num(x)] += 1
            self.cnt_color[id2color(x)] += 1
            self.cnt_num_eachcolor[id2color(x)][id2num(x)] += 1
            self.nums.append(id2num(x))

        self.judge_num_eachcolor = [[] for i in range(4)]

        for i in range(4):
            self.judge_num_eachcolor[i] = list(map(judge_exist, self.cnt_num_eachcolor[i]))


        self.nums.sort(reverse=True)
        for i in range(12, -1, -1):
            if self.cnt_num[i] == 1:
                self.single.append(i)
            elif self.cnt_num[i] == 2:
                self.pair.append(i)
            elif self.cnt_num[i] == 3:
                self.tripple.append(i)
        self.single.sort(reverse=True)
        self.pair.sort(reverse=True)
        self.tripple.sort(reverse=True)

        # calculate the level of the poker hand
        for i in range(4):
            if self.judge_num_eachcolor[i][8:13].count(True) == 5:
                self.level = 10
                return


        for i in range(4):

            for j in range(7, -1, -1):
                if self.judge_num_eachcolor[i][j:j+5].count(True) == 5:
                    self.level = 9
                    self.maxnum = j + 4
                    return
            if self.judge_num_eachcolor[i][12] and self.judge_num_eachcolor[i][:4].count(True) == 4:
                    self.level = 9
                    self.maxnum = 3
                    return



        for i in range(12, -1, -1):
            if self.cnt_num[i] == 4:
                self.maxnum = i
                self.level = 8
                for j in range(4):
                    self.nums.remove(i)
                return


        tripple = self.cnt_num.count(3)
        if tripple > 1:
            self.level = 7
            return
        elif tripple > 0:
            if self.cnt_num.count(2) > 0:
                self.level = 7
                return

        for i in range(4):
            if self.cnt_color[i] >= 5:
                self.nums = []
                for card in cards:
                    if id2color(card) == i:
                        self.nums.append(id2num(card))
                self.nums.sort(reverse=True)
                self.nums = self.nums[:5]
                self.maxnum = self.nums[0]
                self.level = 6
                return

        for i in range(8, -1, -1):
            flag = 1
            for j in range(i, i + 5):
                if self.cnt_num[j] == 0:
                    flag = 0
                    break
            if flag == 1:
                self.maxnum = i + 4
                self.level = 5
                return
        if self.cnt_num[12] and list(map(judge_exist, self.cnt_num[:4])).count(True) == 4:
            self.maxnum = 3
            self.level = 5
            return


        for i in range(12, -1, -1):
            if self.cnt_num[i] == 3:
                self.maxnum = i
                self.level = 4
                self.nums.remove(i)
                self.nums.remove(i)
                self.nums.remove(i)
                self.nums = self.nums[:min(len(self.nums), 2)]
                return


        if self.cnt_num.count(2) > 1:
            self.level = 3
            return


        for i in range(12, -1, -1):
            if self.cnt_num[i] == 2:
                self.maxnum = i
                self.level = 2

                self.nums.remove(i)
                self.nums.remove(i)
                self.nums = self.nums[:min(len(self.nums), 3)]
                return


        if self.cnt_num.count(1) == 7:
            self.level = 1
            self.nums = self.nums[:min(len(self.nums), 5)]
            return

        self.level = -1

    def __str__(self):
        return 'level = %s' % self.level


def cmp(x,y):  # x < y return 1
    if x > y: return -1
    elif x == y: return 0
    else: return 1

# find the bigger of two poker hand(7 cards), if cards0 < cards1 then return 1, cards0 > cards1 return -1, else return 0
def judge_two(cards0, cards1):
    hand0 = Hand(cards0)
    hand1 = Hand(cards1)
    if hand0.level > hand1.level:
        return -1
    elif hand0.level < hand1.level:
        return 1
    else:
        if hand0.level in [5, 9]:
            return cmp(hand0.maxnum, hand1.maxnum)
        elif hand0.level in [1, 2, 4]:
            t = cmp(hand0.maxnum, hand1.maxnum)
            if t == 1: return 1
            elif t == -1: return -1
            else:
                if hand0.nums < hand1.nums:
                    return 1
                elif hand0.nums == hand1.nums:
                    return 0
                else:
                    return -1

        elif hand0.level == 6:
            if hand0.nums < hand1.nums:
                return 1
            elif hand0.nums > hand1.nums:
                return -1
            else:
                return 0

        elif hand0.level == 8:
            t = cmp(hand0.maxnum, hand1.maxnum)
            if t == 1:
                return 1
            elif t == -1:
                return -1
            else:
                return cmp(hand0.nums[0], hand1.nums[0])

        elif hand0.level == 3:
            if cmp(hand0.pair[0], hand1.pair[0]) != 0:
                return cmp(hand0.pair[0], hand1.pair[0])
            elif cmp(hand0.pair[1], hand1.pair[1]) != 0:
                return cmp(hand0.pair[1], hand1.pair[1])
            else:
                hand0.pair = hand0.pair[2:]
                hand1.pair = hand1.pair[2:]
                tmp0 = hand0.pair + hand0.pair + hand0.single
                tmp0.sort(reverse=True)
                tmp1 = hand1.pair + hand1.pair + hand1.single
                tmp1.sort(reverse=True)
                if tmp0[0] < tmp1[0]:
                    return 1
                elif tmp0[0] == tmp1[0]:
                    return 0
                else:
                    return -1

        elif hand0.level == 7:
            if cmp(hand0.tripple[0], hand1.tripple[0]) != 0:
                return cmp(hand0.tripple[0], hand1.tripple[0])
            else:
                tmp0 = hand0.pair
                tmp1 = hand1.pair
                if len(hand0.tripple) > 1:
                    tmp0.append(hand0.tripple[1])
                if len(hand1.tripple) > 1:
                    tmp1.append(hand1.tripple[1])
                tmp0.sort(reverse=True)
                tmp1.sort(reverse=True)
                if tmp0[0] < tmp1[0]:
                    return 1
                elif tmp0[0] == tmp1[0]:
                    return 0
                else:
                    return -1
        else:
            pass
            # assert 0
        return 0

class Player(object):

    def __init__(self, initMoney, state):
        self.active = True      # if the player is active(haven't giveups)
        self.money = initMoney  # money player has
        self.bet = 0            # the bet in this round
        self.cards = []         # private cards
        self.allin = 0          # if the player has all in
        self.totalbet = 0       # the bet in total(all round)
        self.state = state      # state

        ## user data
        self.username = ''

        ## session data
        self.token = ''
        self.connected = False
        self.last_msg_time = None
        self.game_over_sent = False

    # raise the bet by amount
    def raisebet(self, amount):
        self.money -= amount
        self.bet += amount
        assert self.money > 0

    # player allin
    def allinbet(self):
        self.bet += self.money
        self.allin = 1
        self.money = 0

    def getcards(self):
        return self.cards + self.state.sharedcards

    def __str__(self):
        return 'player: active = %s, money = %s, bet = %s, allin = %s' % (self.active, self.money, self.bet, self.allin)



class State(object):
    def __init__(self,  totalPlayer, initMoney, bigBlind):
        ''' class to hold the game '''
        self.totalPlayer = totalPlayer # total players in the game
        self.bigBlind = bigBlind       # bigBlind, every bet should be multiple of smallBlind which is half of bigBlind.
        # self.button = button           # the button position
        self.currpos = 0               # current position
        self.playernum = totalPlayer   # active player number
        self.moneypot = 0              # money in the pot
        self.minbet = bigBlind         # minimum bet to call in this round, total bet
        self.sharedcards = []          # shared careds in the game
        self.turnNum = 0               # 0, 1, 2, 3 for pre-flop round, flop round, turn round and river round
        self.last_raised = bigBlind    # the amount of bet raise last time
        self.player = []               # All players. You can check them to help your decision. The 'cards' field of other player is not visiable for sure.
        for i in range(totalPlayer):
            self.player.append(Player(initMoney, self))

        # self.logger = logger

    def set_user_money(self, initMoney):
        for i in range(self.totalPlayer):
            self.player[i].money = initMoney[i]
            print('user at pos {} has {}'.format(i, self.player[i].money), flush=True)

    def __str__(self):
        return 'state: currpos = %s, playernum = %s, moneypot = %s, minbet = %s, last_raised = %s' \
               % (self.currpos, self.playernum, self.moneypot, self.minbet, self.last_raised)

    def restore(self, turn, button, bigBlind):      # restore the state before each round
        self.turnNum = turn
        self.currpos = button
        self.minbet = 0
        self.last_raised = bigBlind

    def update(self, totalPlayer):                       # update the state after each round
        for i in range(totalPlayer):
            self.player[i].totalbet += self.player[i].bet
            self.player[i].bet = 0

    # judge if the round is over
    def round_over(self):
        if self.playernum == 1:
            return 1
        for i in range(self.totalPlayer):
            if (self.player[i].active is True) and (self.player[i].allin == 0):
                return 0
        for i in range(self.totalPlayer):
            if self.player[i].active is True and (self.player[i].bet != self.minbet and self.player[i].allin == 0):
                return 0
        if self.turnNum != 0 and self.minbet == 0:
            return 0
        return 1

    # calculate the next position
    def nextpos(self, pos):
        self.currpos = (pos + 1) % self.totalPlayer
        return self.currpos


# 没有使用？
class Decision(object):
    giveup = 0   # 弃牌
    allin = 0    # 全押
    check = 0    # 过牌
    callbet = 0  # 跟注
    raisebet = 0 # 加注
    amount = 0   # 本轮中加注到amount

    def clear(self):
        self.giveup = self.allin = self.check = self.callbet = self.raisebet = self.amount = 0

    def update(self, a):
        self.giveup = a[0]
        self.allin = a[1]
        self.check = a[2]
        self.callbet = a[3]
        self.raisebet = a[4]
        self.amount = a[5]

    def isValid(self):
        # 只能进行一项操作
        if self.giveup + self.allin + self.check + self.callbet + self.raisebet == 1:
            if self.raisebet == 1 and self.amount == 0:
                return False
            return True
        return False

    def fix(self):
        amount = self.amount
        setname = ''
        # 
        for k, v in self.__dict__.items():
            # 寻找有效操作标记
            if v == 1 and k != 'amount':
                setname = k
            setattr(self, k, 0)
        # 未找到有效操作直接记为弃牌
        if setname == '':
            setattr(self, 'giveup', 1)
        else:
            setattr(self, setname, 1)
            if setname == 'raisebet':
                if amount != 0:
                    setattr(self, 'amount', amount)
                else:
                    setattr(self, 'callbet', 1)
                    setattr(self, 'raisebet', 0)


    def __str__(self):
        # 返回全部操作标记
        return 'giveup=%s, allin=%s, check=%s, callbet=%s, raisebet=%s, amount=%s' % (self.giveup,self.allin,self.check,
                                                                            self.callbet, self.raisebet, self.amount)