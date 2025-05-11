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
    minbet=max( BetLimit[1] ,0) # 许多信息会反映到minbet中，所以可能不需要回溯上一个选手的action
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
        # 两张牌
        if (cards[0] // 4) != (cards[1] // 4): # 非对子
            if max(cards) < 32:
                # 最大不超过9：若跟注后超过100，放弃。否则跟注
                if totalbet <= 50:
                    decision["callbet"] = 1
                else:
                    decision["giveup"] = 1
            if max(cards) < 44:
                # 最大为10-Q：若跟注后超过150，放弃。否则跟注
                if totalbet <= 100:
                    decision["callbet"] = 1
                else:
                    decision["giveup"] = 1
            else:
                # 最大为K-A： 若跟注后超过200，放弃。否则跟注
                if totalbet <= 150:
                    decision["callbet"] = 1
                else:
                    decision["giveup"] = 1
        else:
            # 对子
            if max(cards) < 44:
                # 对子，不超过Q：跟注。若跟注后低于200，加注到200以上
                if totalbet < 100:
                    decision = add_bet(table_info, 105)
                else:
                    decision["callbet"] = 1
            else:
                # 双A、双K：跟注。若跟注后低于300，加注到300
                if totalbet < 150:
                    decision = add_bet(table_info, 155)
                else:
                    decision["callbet"] = 1

    elif num == 5: # 五张牌
        if sum < 4:
            # 直接放弃
            decision["giveup"] = 1
        elif sum >= 4 and sum < 10:
            # 若跟注后超过150，放弃。否则跟注
            # 若已下的注额大于200, 且本次需跟注额不大于50， 则跟注
            if minbet == 0:
                decision['check'] = 1
            elif totalbet < 150:
                decision["callbet"] = 1
            elif totalbet > 200 and delta < 50:
                decision["callbet"] = 1
            elif id == small_b or id == big_b:
                decision["callbet"] = 1
                decision["giveup"] = 0
                decision["raisebet"] = 0
            else:
                decision["giveup"] = 1
            if delta > handchips:
                decision["callbet"] = 0
                decision["giveup"] = 1
                decision["raisebet"] = 0

        elif sum >= 10 and sum < 20:
            # 跟注。若跟注后低于300，加注到300
            if totalbet < 200:
                decision = add_bet(table_info, 205)
            else:
                decision["callbet"] = 1
        elif sum >= 20 and sum < 50:
            # 跟注。若跟注后低于600，加注到600
            if totalbet < 300:
                decision = add_bet(table_info, 305)
                if decision['amount'] > handchips:
                    decision.allin = 1
            else:
                decision["callbet"] = 1
                if delta > handchips:
                    decision.allin = 1 # 这里的含义是，牌还不错但钱不多的时候，可以一搏
        else:
            decision["callbet"] = 1
            if delta > handchips:
                decision.allin = 1
    
        if BetLimit[0] == 0:
            decision['check'] = 1
            decision["callbet"] = 0
            decision["giveup"] = 0
            decision["raisebet"] = 0

    elif num == 6: # 六张牌
        if sum < 2:
            # 直接放弃
            decision["giveup"] = 1
        elif sum >= 2 and sum < 8:
            # 若跟注后超过300，放弃。否则跟注
            # 若已下的注额大于200, 且本次需跟注额不大于50， 则跟注
            if minbet == 0:
                decision['check'] = 1
            elif totalbet < 200:
                    decision["callbet"] = 1
            elif totalbet > 200 and delta < 40:
                decision["callbet"] = 1
            else:
                decision["giveup"] = 1

                
        elif sum >= 8 and sum < 20:
            # 跟注。若跟注后低于300，加注到300
            if totalbet < 300:
                decision = add_bet(table_info, 305)
                if decision['amount'] > handchips:
                    decision.allin = 1
            else:
                decision["callbet"] = 1
                if delta > handchips:
                    decision.allin = 1

        elif sum >= 20 and sum < 40:
            # 跟注。若跟注后低于600，加注到600
            if totalbet < 400:
                if id == dealer:
                    decision = add_bet(table_info, 605)
                else:
                    decision = add_bet(table_info, 405)
                if decision['amount'] > handchips:
                    decision.allin = 1
            else:
                decision["callbet"] = 1
                if delta > handchips:
                    decision.allin = 1
        else:
            # allin
            decision.allin = 1

        if BetLimit[0] == 0:
            decision['check'] = 1
            decision["callbet"] = 0
            decision["giveup"] = 0
            decision["raisebet"] = 0
            decision.allin = 1

    elif num == 7:
        # 七张牌
        if level == 7:
            # allin
            decision.allin = 1
        elif level == 6:
            # 跟注，若跟注后低于600，加注到600
            if totalbet < 400:
                if id == dealer:
                    decision = add_bet(table_info, 605)
                else:
                    decision = add_bet(table_info, 405)
                if decision['amount'] > handchips:
                    decision.allin = 1
            else:
                decision["callbet"] = 1
                if delta > handchips:
                    decision.allin = 1

        elif level == 5:
            # 跟注，若跟注后低于500，加注到500
            if totalbet < 400:
                decision = add_bet(table_info, 405)
                if decision['amount'] > handchips:
                    decision.allin = 1
            else:
                decision["callbet"] = 1
                if delta > handchips:
                    decision.allin = 1

        elif level == 4:
            # 跟注，若跟注后低于400，加注到400
            if totalbet < 400:
                decision = add_bet(table_info, 405)
            else:
                decision["callbet"] = 1

        elif level == 3:
            # 若跟注后超过500，放弃。否则跟注。若跟注后低于300，加注到300
            # 若已下的注额大于200, 且本次需跟注额不大于50， 则跟注
            if minbet == 0:
                decision['check'] = 1
            elif totalbet < 200:
                decision = add_bet(table_info, 205)
            elif totalbet < 300:
                decision["callbet"] = 1
            elif totalbet > 200 and delta < 50:
                decision["callbet"] = 1
            else:
                decision["giveup"] = 1
        elif level == 2:
            if minbet == 0:
                decision['check'] = 1
            elif cards.count(0) == 2 or cards.count(12) == 2:
                # 双A双K 若跟注后超过200，放弃。否则跟注
                # 若已下的注额大于200, 且本次需跟注额不大于50， 则跟注
                if totalbet < 200:
                    decision["callbet"] = 1
                elif totalbet > 200 and delta < 50:
                    decision["callbet"] = 1
                else:
                    decision["giveup"] = 1
            else:
                # 不超过双Q 若跟注后超过200，放弃。否则跟注
                if totalbet > 200:
                    decision["giveup"] = 1
                else:
                    decision["callbet"] = 1
        elif level == 1:
            decision["giveup"] = 1
        else:
            print('the num of cards is {}'.format(num))
            assert(0)

    if decision["callbet"] == 1 and BetLimit[0] == 0:
        t = random.randint(0,2)
        if t == 0:
            decision["callbet"] = 0
            decision["raisebet"] = 1
            decision.allin = 0
            decision['check'] = 0
            decision["callbet"] = 0
            decision["amount"] =table_info['RoomSetting']['BB']

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