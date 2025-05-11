from pywebio import *
from pywebio.input import *
from pyecharts.charts import HeatMap
from pyecharts.charts import Pie
import random
import time
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.faker import Faker
import numpy

# Initial: set all suits and ranks and cards
all_suits_name = ["黑桃 ♠", "红桃 ♥", "方片 ♦", "梅花 ♣"]
all_suits = [0, 1, 2, 3]
all_ranks_name = ['A', 'K', 'Q', 'J', '10', '9', '8', '7',
                  '6', '5', '4', '3', '2']
all_ranks = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
all_hands_rankings = ["Straight Flush",
                      "Four of a Kind",
                      "Full House",
                      "Flush",
                      "Straight",
                      "Three of a Kind",
                      "Two Pair",
                      "Pair",
                      "High Card"]

all_hands_rankings_name = ["高牌",
                           "一对",
                           "两对",
                           "三条",
                           "顺子",
                           "同花",
                           "葫芦",
                           "金刚",
                           "花顺"]

all_cards = []


def rank_point(rank):
    return 14 - all_ranks_name.index(rank)

def suit_point(suit):
    return all_suits_name.index(suit) 


for suit in all_suits:
    for rank in all_ranks:
        all_cards.append((rank, suit))

# print(str(all_cards))


def key_card_ranks(a_card):
    return a_card[0] * 100 + a_card[1]


def key_card_suits(a_card):
    return a_card[1] * 100 + a_card[0]


def get_hand_power(a_hand):
    if(len(a_hand) != 7):
        print("hand_len ERROR :{}, which should be 7".format(len(a_hand)))
        return None
    # check if Straight Flush
    # power :[8,min straight rank]
    a_hand.sort(key=key_card_suits, reverse=True)
    cur_suit = -1
    has_Ace = False
    last_rank = -1
    cur_len = 0
    for i in range(0, len(a_hand)):
        # check if new suit
        if(a_hand[i][1] != cur_suit):
            if(cur_len == 4 and last_rank == 2 and has_Ace):
                # bingo
                return [8, 1]
            cur_suit = a_hand[i][1]
            if(a_hand[i][0] == 14):
                # has Ace
                has_Ace = True
            else:
                has_Ace = False
            last_rank = a_hand[i][0]
            cur_len = 1
        else:
            if(a_hand[i][0] == last_rank - 1):
                cur_len = cur_len + 1
                if(cur_len == 5):
                    # bingo
                    return [8, a_hand[i][0]]
            else:
                cur_len = 1
            last_rank = a_hand[i][0]
    if(cur_len == 4 and last_rank == 2 and has_Ace):
        # bingo
        return [8, 1]

    # check if Four of a Kind
    # power :[7,four of kind's rank,other hight card's rank ]
    a_hand.sort(key=key_card_ranks, reverse=True)
    cur_rank = -1
    cur_rep = 0
    for i in range(0, len(a_hand)):
        if(a_hand[i][0] == cur_rank):
            cur_rep = cur_rep + 1
            if(cur_rep == 4):
                # bingo,and to get the other one card
                for i2 in range(0, len(a_hand)):
                    if(a_hand[i2][0] != cur_rank):
                        return [7, cur_rank, a_hand[i2][0]]
        else:
            cur_rep = 1
            cur_rank = a_hand[i][0]

    # check if Full House
    # power :[6,three kind rank,highest pair rank]
    three_kind_rank = -1
    pair_rank = -1
    cur_rank = -1
    cur_rep = 0
    for i in range(0, len(a_hand)):
        if(a_hand[i][0] == cur_rank):
            cur_rep = cur_rep + 1
        else:
            if(cur_rep == 3 and cur_rank > three_kind_rank):
                three_kind_rank = cur_rank
            else:
                if(cur_rep == 2 and cur_rank > pair_rank):
                    pair_rank = cur_rank
            cur_rank = a_hand[i][0]
            cur_rep = 1
    if(three_kind_rank != -1 and pair_rank != -1):
        # bingo
        return [6, three_kind_rank, pair_rank]

    # check if Flush
    # power :[5,1-5 rank of the flush....]
    a_hand.sort(key=key_card_suits, reverse=True)
    suit_first_index = -1
    cur_suit = -1
    cur_len = 0
    for i in range(0, len(a_hand)):
        if(a_hand[i][1] == cur_suit):
            cur_len = cur_len + 1
            if(cur_len == 5):
                # bingo
                return [5, a_hand[suit_first_index][0],
                        a_hand[suit_first_index + 1][0],
                        a_hand[suit_first_index + 2][0],
                        a_hand[suit_first_index + 3][0],
                        a_hand[suit_first_index + 4][0]]
        else:
            cur_len = 1
            cur_suit = a_hand[i][1]
            suit_first_index = i

    # check if Straight
    # power :[4,min straight rank]
    a_hand.sort(key=key_card_ranks, reverse=True)
    last_rank = -1
    cur_len = 0
    has_Ace = False
    if(a_hand[0][0] == 14):
        has_Ace = True
    for i in range(0, len(a_hand)):
        if(a_hand[i][0] == last_rank):
            continue
        if(a_hand[i][0] == last_rank - 1):
            cur_len = cur_len + 1
            last_rank = a_hand[i][0]
            if(cur_len == 5):
                # bingo
                return [4, last_rank]
            if (cur_len == 4 and a_hand[i][0] == 2 and has_Ace):
                # bingo
                return [4, 1]
        else:
            cur_len = 1
            last_rank = a_hand[i][0]

    # check if Three of a Kind
    # power :[3,three kind rank,other two high,,]
    first_high = -1
    if(three_kind_rank != -1):
        for i in range(0, len(a_hand)):
            if(first_high == -1 and a_hand[i][0] != three_kind_rank):
                first_high = a_hand[i][0]
            else:
                if(first_high != -1
                   and a_hand[i][0] != three_kind_rank
                   and a_hand[i][0] != first_high):
                    return [3, three_kind_rank, first_high, a_hand[i][0]]

    # check if Two Pair
    # power :[2,Top Pair rank,Second Pair Rank,High Rank]
    if(pair_rank != -1):
        top_pair_rank = -1
        for i in range(0, len(a_hand)):
            if(i != len(a_hand) - 1 and a_hand[i][0] == a_hand[i + 1][0]):
                if(top_pair_rank == -1):
                    top_pair_rank = a_hand[i][0]
                else:
                    for i2 in range(0, len(a_hand)):
                        if(a_hand[i2][0] != top_pair_rank
                           and a_hand[i2][0] != a_hand[i][0]):
                            return [2, top_pair_rank,
                                    a_hand[i][0], a_hand[i2][0]]

    # check if Pair
    # power :[1,Top Pair rank,First High Rank,Second High Rank,Third High Rank]
    if(pair_rank != -1):
        first_high = -1
        second_high = -1
        for i in range(0, len(a_hand)):
            if(a_hand[i][0] == pair_rank):
                continue
            if(first_high == -1):
                first_high = a_hand[i][0]
                continue
            if(second_high == -1):
                second_high = a_hand[i][0]
                continue
            return [1, pair_rank, first_high, second_high, a_hand[i][0]]

    # check if High Card
    # power :[0,Top Pair rank,First High Rank,Second High Rank,Third High Rank]
    return [0, a_hand[0][0], a_hand[1][0],
            a_hand[2][0], a_hand[3][0],
            a_hand[4][0], a_hand[5][0]]


power_coefficient = [pow(16, 6), pow(16, 5), pow(16, 4),
                     pow(16, 3), pow(16, 2), pow(16, 1), 1]


def key_hand_power(hand_power):
    result = 0
    for i in range(0, len(hand_power)):
        result = result + hand_power[i] * power_coefficient[i]
    return result


# Random Deal Test
def RandomDeal(player_num, player_cards, desk_cards):
    left_cards = all_cards.copy()
    for i in range(0, player_num):
        for i2 in range(0, 2):
            if(player_cards[i][i2] is not None):
                left_cards.remove(player_cards[i][i2])

    test_times = 20000
    win_times = 0
    tied_times = 0
    lose_times = 0
    
    hands_rankings = numpy.zeros([player_num,9], dtype=int)
    win_data = numpy.zeros([player_num,3], dtype=int)
    hands_rankings_win_data = numpy.zeros([player_num,3,9], dtype=int)
    rights_of_po = numpy.zeros([player_num,3,9], dtype=int)

    for cur_test_time in range(0, test_times):
        left_cards_test = [y for y in left_cards]
        player_cards_test = [[x for x in y] for y in player_cards]
        desk_cards_test = [y for y in desk_cards]
        random.shuffle(left_cards_test)
        deal_at = 0
        for i in range(0, player_num):
            for i2 in range(0, 2):
                if(player_cards_test[i][i2] is None):
                    player_cards_test[i][i2] = left_cards_test[deal_at]
                    deal_at = deal_at + 1

        for i in range(0, 5):
            if(desk_cards_test[i] is None):
                desk_cards_test[i] = left_cards_test[deal_at]
                deal_at = deal_at + 1

        player_hands = []
        player_hands_power = []
        player_key_hands_power = []
        for i in range(0, player_num):
            player_hands.append(desk_cards_test + player_cards_test[i])
            
            player_hands_power.append(get_hand_power(player_hands[i]))
            
            player_key_hands_power.append(key_hand_power(player_hands_power[i]))

        most_key_power = player_key_hands_power[0]
        is_tied = False
        n_winner = 0
        for i in range(1, player_num):
            if(player_key_hands_power[i] > most_key_power):
                most_key_power = player_key_hands_power[i]
                
        for i in range(0, player_num):
            if(player_key_hands_power[i] == most_key_power):
                n_winner = n_winner + 1
                
        if(n_winner > 1):
            is_tied = True
                
        for i in range(0, player_num):
            hands_rankings[i,player_hands_power[i][0]] = hands_rankings[i,player_hands_power[i][0]] + 1
            if(player_key_hands_power[i] == most_key_power):
                if(is_tied):
                    win_data[i][1] = win_data[i][1] + 1
                    hands_rankings_win_data[i][1][player_hands_power[i][0]] = hands_rankings_win_data[i][1][player_hands_power[i][0]] + 1
                else:
                    win_data[i][0] = win_data[i][0] + 1
                    hands_rankings_win_data[i][0][player_hands_power[i][0]] = hands_rankings_win_data[i][0][player_hands_power[i][0]] + 1
            else:
                win_data[i][2] = win_data[i][2] + 1
                hands_rankings_win_data[i][2][player_hands_power[i][0]] = hands_rankings_win_data[i][2][player_hands_power[i][0]] + 1

    result = {}
    result['hands_rankings'] = numpy.around(hands_rankings*1000.0/test_times)/1000.0
    result['win_rate'] = numpy.around(win_data*1000.0/test_times)/1000.0
    result['hands_rankings_win_rate'] = numpy.around(hands_rankings_win_data*1000.0/test_times)/1000.0
    return result



def card_repeat_check(data):
    choosen_card_names = []
    for key in data:
        if(data[key] not in ['随机','----------------']):
            if(data[key] in choosen_card_names):
                return (key, '选牌重复')
            else:
                choosen_card_names.append(data[key])
            
    

total_visit_times = 0
total_run_times = 0

def main():  # PyWebIO application function
    global total_visit_times
    total_visit_times = total_visit_times + 1
    output.put_text("total visit times: {}".format(total_visit_times))
    player_num = select('选择玩家数量', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    all_card_names = ['随机']
    all_card_rank = [None]
    all_card_suit = [None]
    for suit in all_suits_name:
        for rank in all_ranks_name:
            all_card_names.append("{}{}".format(suit, rank))
            all_card_rank.append(rank_point(rank))
            all_card_suit.append(suit_point(suit))
        all_card_names.append("----------------")
        all_card_rank.append(0)
        all_card_suit.append(0)

    input_content = []
    for i in range(0, player_num):
        input_content.append(select("#{} 玩家的第 1 张牌".format(i),all_card_names,
                                    name="player_{}_card_1".format(i)))
        input_content.append(select("#{} 玩家的第 2 张牌".format(i),all_card_names,
                                    name="player_{}_card_2".format(i)))

    hand_data = input_group("玩家手牌设置", input_content, validate=card_repeat_check)
    player_cards = []
    for i in range(0, player_num):
        cur_player_cards = []
        for i2 in range(1,3):
            if(hand_data["player_{}_card_{}".format(i, i2)] in ['随机','----------------']):
                cur_player_cards.append(None)
            else:
                cur_player_cards.append((all_card_rank[all_card_names.index(hand_data["player_{}_card_{}".format(i, i2)])],
                                         all_card_suit[all_card_names.index(hand_data["player_{}_card_{}".format(i, i2)])]))
                del all_card_rank[all_card_names.index(hand_data["player_{}_card_{}".format(i, i2)])]
                del all_card_suit[all_card_names.index(hand_data["player_{}_card_{}".format(i, i2)])]
                all_card_names.remove(hand_data["player_{}_card_{}".format(i, i2)])
        player_cards.append(cur_player_cards)
                                               
    input_content = []
    for i in range(0, 5):
        input_content.append(select("第 {} 张公共牌".format(i),all_card_names,
                                    name="desk_card_{}".format(i)))
        
    desk_data = input_group("公共牌设置", input_content, validate=card_repeat_check)

    desk_cards = []
    for i in range(0, 5):
        if(desk_data["desk_card_{}".format(i)] in ['随机','----------------']):
            desk_cards.append(None)
        else:
            desk_cards.append((all_card_rank[all_card_names.index(desk_data["desk_card_{}".format(i)])],
                               all_card_suit[all_card_names.index(desk_data["desk_card_{}".format(i)])]))
            
    global total_run_times
    total_run_times = total_run_times + 1
    
    output.put_text("already run for {} times".format(total_run_times))
    output.put_text("player_cards_names:", str(hand_data))
    output.put_text("desk_cards_names:", str(desk_data))
    output.put_text("player_cards:", str(player_cards))
    output.put_text("desk_cards:", str(desk_cards))
    output.put_text("running....".format(total_run_times))
    
    start = time.time()
    result = RandomDeal(player_num, player_cards, desk_cards)
    end = time.time()

    output.put_text("finished.cost {} seconds".format(str(end - start)))
    

    
    # 所有玩家成牌胜负
    #     output.put_text("win_rate:", str(result['win_rate']))
    output.put_html("<h3>各玩家输赢</h3>")
    players_win_rate_0 = [float(result['win_rate'][i][0]) for i in range(player_num)]
    players_win_rate_1 = [float(result['win_rate'][i][1]) for i in range(player_num)]
    players_win_rate_2 = [float(result['win_rate'][i][2]) for i in range(player_num)]
    bar = (
        Bar(init_opts=opts.InitOpts(width="350px"))
        .add_xaxis(["#{}".format(i) for i in range(player_num)])
        .add_yaxis("赢", players_win_rate_0, stack = "stack1") # y轴设置
        .add_yaxis("平", players_win_rate_1, stack = "stack1") # y轴设置
        .add_yaxis("输", players_win_rate_2, stack = "stack1") # y轴设置
        .set_global_opts(title_opts=opts.TitleOpts(title="", subtitle=""))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False)) 
        .reversal_axis()
        )

    output.put_html(bar.render_notebook())
    
    # 各玩家成牌分布
    output.put_html("<h3>各玩家成牌分布</h3>")
    hands_rankings = numpy.round(result['hands_rankings'] * 1000.0)/10.0
    hands_rankings_for_output = [[i,j,hands_rankings[i][j]] for i in range(player_num) for j in range(9)]

    heat = (HeatMap(init_opts=opts.InitOpts(width="350px"))
            .add_xaxis(["#{}".format(i) for i in range(player_num)])
            .add_yaxis("比例",
                       all_hands_rankings_name,
                       hands_rankings_for_output,
                       label_opts=opts.LabelOpts(is_show=True, position="inside"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="", subtitle=" "),
                visualmap_opts=opts.VisualMapOpts(is_show=False),
                legend_opts=opts.LegendOpts(is_show=False))
           )

    output.put_html(heat.render_notebook())
    
    # player 0 成牌数据
    cate = all_hands_rankings_name
    data = result['hands_rankings'][0]
    output.put_html("<h3>#0 玩家成牌</h3>")
    pie = (Pie(init_opts=opts.InitOpts(width="350px"))
           .add('', [list(z) for z in zip(cate, data)],
                radius=["30%", "75%"],
                rosetype="radius")
           .set_global_opts(title_opts=opts.TitleOpts(title="", subtitle=" "))
           .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
          )

    output.put_html(pie.render_notebook())
    
    # player 0 胜负数据
    cate = ['赢', '平', '输']
    data = result['win_rate'][0]

    output.put_html("<h3>#0 玩家输赢</h3>")
    pie = (Pie(init_opts=opts.InitOpts(width="350px"))
           .add('', [list(z) for z in zip(cate, data)],
                radius=["30%", "75%"],
                rosetype="radius")
           .set_global_opts(title_opts=opts.TitleOpts(title="", subtitle=" "))
           .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
          )

    output.put_html(pie.render_notebook())
    
    # player 0 成牌胜负
    output.put_html("<h3>#0 玩家成牌输赢</h3>")
    hands_rankings_win_rate_0 = [float(result['hands_rankings_win_rate'][0][0][i]) for i in range(9)]
    hands_rankings_win_rate_1 = [float(result['hands_rankings_win_rate'][0][1][i]) for i in range(9)]
    hands_rankings_win_rate_2 = [float(result['hands_rankings_win_rate'][0][2][i]) for i in range(9)]
    bar = (
        Bar(init_opts=opts.InitOpts(width="350px"))
        .add_xaxis(all_hands_rankings_name)
        .add_yaxis("赢", hands_rankings_win_rate_0, stack = "stack1") # y轴设置
        .add_yaxis("平", hands_rankings_win_rate_1, stack = "stack1") # y轴设置
        .add_yaxis("输", hands_rankings_win_rate_2, stack = "stack1") # y轴设置
        .set_global_opts(title_opts=opts.TitleOpts(title="", subtitle=""))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False)) 
        .reversal_axis()
        )

    output.put_html(bar.render_notebook())

    
config(title='计算器')
start_server(main, port=8889, debug=True)