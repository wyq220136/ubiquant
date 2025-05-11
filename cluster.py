"""
    实现对手牌进行评估的函数，将连续问题离散化
"""
from itertools import combinations
from collections import Counter

class HandEvaluator:
    def __init__(self, public_cards: list, private_cards: list):
        self.private_cards = private_cards
        self.public_cards = public_cards
        self.all_cards = public_cards + private_cards
    
    def evaluate_hand(self):
        if len(self.public_cards) == 0:
            # 公共手牌为0张时，只有私牌
            hand_strength = self.private_cards, self.hand_value_private(self.private_cards)
        else:
            # 公共手牌大于0张时，计算私牌与公共牌的所有组合
            hand_strength = self.evaluate_best_hand(self.all_cards)
        return hand_strength
    
    def evaluate_best_hand(self, available_cards):
        best_hand = None
        best_value = -1
        
        # Generate all possible 5-card hands from the available cards
        for hand in combinations(available_cards, 5):  
            value = self.hand_value(hand)
            if value > best_value:
                best_value = value
                best_hand = hand
        
        return best_hand, best_value
    
    def hand_value(self, hand):
        """ Evaluate the value of a given hand """
        suits = [card[1] for card in hand]
        values = [card[0] for card in hand]
        
        # Validate card ranks (only keep valid values)
        valid_values = '23456789TJQKA'
        values = [v for v in values if v in valid_values]
        
        value_counts = Counter(values)
        suit_counts = Counter(suits)
        
        is_flush = len(suit_counts) == 1
        is_straight = self.is_straight(values)
        
        if is_straight and is_flush:
            if sorted(values) == ['T', 'J', 'Q', 'K', 'A']:
                return 10  # Royal Flush (This is a special Straight Flush)
            return 9  # Straight Flush
        elif 4 in value_counts.values():
            return 8  # Four of a Kind
        elif 3 in value_counts.values() and 2 in value_counts.values():
            return 7  # Full House
        elif is_flush:
            return 6  # Flush
        elif is_straight:
            return 5  # Straight
        elif 3 in value_counts.values():
            return 4  # Three of a Kind
        elif list(value_counts.values()).count(2) == 2:
            return 3  # Two Pair
        elif 2 in value_counts.values():
            return 2  # One Pair
        else:
            return 1  # High Card
    
    # def hand_value_private(self, hand):
    #     """ Evaluate the value of a given hand with customized scoring """
    #     suits = [card[1] for card in hand]
    #     values = [card[0] for card in hand]
        
    #     # Validate card ranks (only keep valid values)
    #     valid_values = '23456789TJQKA'
    #     values = [v for v in values if v in valid_values]
        
    #     # Count frequency of each card value and suit
    #     value_counts = Counter(values)
    #     suit_counts = Counter(suits)
        
    #     # Check if all cards are of the same suit (flush)
    #     is_flush = len(suit_counts) == 1
        
    #     # Check for adjacent (straight) cards
    #     is_consecutive = self.is_consecutive(values)
        
    #     score = 0
        
    #     # 1. Check for pair and assign value based on the pair's rank
    #     if len(value_counts) == 4:  # One pair
    #         pair_value = max(value_counts, key=value_counts.get)
    #         score = ord(pair_value) - ord('2') + 2  # Assign value based on rank of the pair
        
    #     # 2. Check if all cards are the same suit (flush)
    #     if is_flush:
    #         score = max(score, 6)  # Same suit, assign 6 points
        
    #     # 3. Check if we have a pair and all cards are of the same suit
    #     if len(value_counts) == 4 and is_flush:
    #         score = max(score, 10)  # Pair and flush, assign 10 points
        
    #     # 4. Check for consecutive cards (straight)
    #     if is_consecutive:
    #         score = max(score, 5)  # Assign 5 points for consecutive cards
        
    #     # 5. Check for consecutive cards and flush (straight flush)
    #     if is_consecutive and is_flush:
    #         score = max(score, 8)  # Assign 8 points for straight flush (consecutive and same suit)
        
    #     return score

    def is_consecutive(self, values):
        """ Check if the cards are consecutive (part of a straight) """
        value_ranks = '23456789TJQKA'
        sorted_values = sorted(set(values), key=lambda v: value_ranks.index(v))
        
        # Handle Ace as both high and low in a straight
        if len(sorted_values) == 5:
            for i in range(len(sorted_values) - 1):
                if value_ranks.index(sorted_values[i+1]) - value_ranks.index(sorted_values[i]) != 1:
                    return False
            return True
        elif sorted_values == ['2', '3', '4', '5', 'A']:  # Special case for Ace-low straight
            return True
        return False

    def is_straight(self, values):
        """ Check if the hand is a straight """
        value_ranks = '23456789TJQKA'
        sorted_values = sorted(set(values), key=lambda v: value_ranks.index(v))
        
        # Handle Ace as both high and low in a straight
        if len(sorted_values) == 5:
            for i in range(len(sorted_values) - 1):
                if value_ranks.index(sorted_values[i+1]) - value_ranks.index(sorted_values[i]) != 1:
                    return False
            return True
        elif sorted_values == ['2', '3', '4', '5', 'A']:  # Special case for Ace-low straight
            return True
        return False

# Example usage:
public_cards = []  # Example of full board
private_cards = ['Kh', 'Kh']

hand_evaluator = HandEvaluator(public_cards, private_cards)
best_hand, hand_strength = hand_evaluator.evaluate_hand()
print("Best Hand:", best_hand)
print("Hand Strength:", hand_strength)
