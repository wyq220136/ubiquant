import random
import itertools
from collections import Counter

class HandEvaluator:
    def __init__(self, public_cards: list, private_cards: list):
        self.public = public_cards
        self.private = private_cards

    def evaluate(self, num_opponents=1, simulations=5000):
        deck = [r + s for r in '23456789TJQKA' for s in 'cdhs'
                if r + s not in self.public + self.private]
        win = 0
        rank_total = 0
        for _ in range(simulations):
            d = deck[:]
            random.shuffle(d)
            public = self.public[:]
            need = 5 - len(public)
            public += d[:need]
            idx = need
            opps = [d[idx + 2*i: idx + 2*i + 2] for i in range(num_opponents)]
            my_rank = self._best_rank(self.private, public)
            opp_ranks = [self._best_rank(o, public) for o in opps]
            if all(my_rank >= orank for orank in opp_ranks):
                win += 1
            rank_total += my_rank
        win_rate = win / simulations
        avg_rank = rank_total / simulations
        score = win_rate * 100 + avg_rank  # combine equity and raw rank
        return {'win_rate': win_rate, 'avg_rank': avg_rank, 'score': score}

    def _best_rank(self, priv, public):
        best = 0
        cards = priv + public
        for combo in itertools.combinations(cards, 5):
            v = self._hand_value(combo)
            if v > best:
                best = v
        return best

    def _hand_value(self, hand):
        suits = [c[1] for c in hand]
        vals = [c[0] for c in hand]
        vc = Counter(vals)
        is_flush = len(set(suits)) == 1
        is_straight = self._is_straight(vals)
        if is_flush and is_straight:
            return 10 if set(vals)==set(['T','J','Q','K','A']) else 9
        if 4 in vc.values():        return 8
        if 3 in vc.values() and 2 in vc.values(): return 7
        if is_flush:               return 6
        if is_straight:            return 5
        if 3 in vc.values():       return 4
        if list(vc.values()).count(2)==2: return 3
        if 2 in vc.values():       return 2
        return 1

    def _is_straight(self, vals):
        order = '23456789TJQKA'
        idxs = sorted({order.index(v) for v in vals if v in order})
        for i in range(len(idxs)-4):
            if idxs[i+4]-idxs[i]==4: return True
        if set(['A','2','3','4','5']).issubset(vals): return True
        return False
    def _hand_value(self, hand):
        order = '23456789TJQKA'
        value_map = {r: i for i, r in enumerate(order, start=2)}
        ranks = sorted([value_map[c[0]] for c in hand], reverse=True)
        suits = [c[1] for c in hand]
        counts = Counter(ranks)
        freqs = sorted(((v, k) for k, v in counts.items()), reverse=True)  # (freq, rank)

        is_flush = len(set(suits)) == 1
        is_straight, high = self._is_straight(ranks)

        if is_flush and is_straight:
            return (9, high)  # Straight flush
        if freqs[0][0] == 4:
            return (8, freqs[0][1], freqs[1][1])  # Four of a kind
        if freqs[0][0] == 3 and freqs[1][0] == 2:
            return (7, freqs[0][1], freqs[1][1])  # Full house
        if is_flush:
            return (6, *ranks)  # Flush
        if is_straight:
            return (5, high)  # Straight
        if freqs[0][0] == 3:
            kickers = [x for x in ranks if x != freqs[0][1]]
            return (4, freqs[0][1], *kickers[:2])  # Three of a kind
        if freqs[0][0] == 2 and freqs[1][0] == 2:
            pair1, pair2 = sorted([freqs[0][1], freqs[1][1]], reverse=True)
            kicker = max([r for r in ranks if r not in [pair1, pair2]])
            return (3, pair1, pair2, kicker)  # Two pair
        if freqs[0][0] == 2:
            kickers = [r for r in ranks if r != freqs[0][1]]
            return (2, freqs[0][1], *kickers[:3])  # One pair
        return (1, *ranks)  # High card

    def _is_straight(self, ranks):
        ranks = sorted(set(ranks), reverse=True)
        if len(ranks) < 5:
            return False, None
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i + 4] == 4:
                return True, ranks[i]
        if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
            return True, 5
        return False, None

# —— 测试示例 ——
# if __name__ == "__main__":
#     tests = [
#         ([], ['Ah', 'Kh']),                    # pre-flop
#         (['2c','7d','Ts'], ['Ah','Kh']),       # flop
#         (['2c','7d','Ts','Jc'], ['Ah','Kh']),  # turn
#         (['2c','7d','Ts','Jc','5h'], ['Ah','Kh']) # river
#     ]
#     for public, private in tests:
#         he = HandEvaluator(public, private)
#         res = he.evaluate(num_opponents=2, simulations=2000)
#         print(f"Public: {public}, Private: {private}")
#         print(f" WinRate: {res['win_rate']:.3f}, AvgRank: {res['avg_rank']:.2f}, Score: {res['score']:.1f}\n")
