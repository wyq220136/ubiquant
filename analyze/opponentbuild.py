
class OpponentRangeEstimator:
    def __init__(self):
        self.opponent_range = {}
        self.opponent_range_history = deque(maxlen=self.history_window)
        self.history_window = 20
        self.opponent_range_history_window = 20
        self.opponent_range_history_window_2 = 20
        self.opponent_range_history_window_3 = 20