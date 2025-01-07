from typing import Dict, List
from enum import Enum

class PatternType(Enum):
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"

class ActionType(Enum):
    LONG = 0
    SHORT = 1
    HOLD = 2

class PatternConfig:
    def __init__(self):
        self.patterns = {
            PatternType.DOUBLE_BOTTOM: {
                "window_size": 20,
                "min_depth": 0.02,  # 2% minimum depth for valley
                "max_deviation": 0.01,  # 1% maximum deviation between bottoms
                "action": ActionType.LONG
            },
            PatternType.DOUBLE_TOP: {
                "window_size": 20,
                "min_height": 0.02,
                "max_deviation": 0.01,
                "action": ActionType.SHORT
            }
        }
        
    def add_pattern_strategy(self, pattern_type: PatternType, config: Dict):
        """Add new pattern strategy"""
        self.patterns[pattern_type] = config
        
    def get_pattern_config(self, pattern_type: PatternType) -> Dict:
        """Get configuration for specific pattern"""
        return self.patterns.get(pattern_type, None)
        
    def get_all_patterns(self) -> List[PatternType]:
        """Get list of all configured patterns"""
        return list(self.patterns.keys())
        
    @staticmethod
    def create_default_strategy():
        """Create default trading strategy configuration"""
        return {
            "entry_conditions": {
                PatternType.DOUBLE_BOTTOM: ActionType.LONG,
                PatternType.DOUBLE_TOP: ActionType.SHORT
            },
            "exit_conditions": {
                PatternType.DOUBLE_TOP: ActionType.SHORT,
                PatternType.DOUBLE_BOTTOM: ActionType.LONG
            },
            "risk_management": {
                "stop_loss": 0.02,  # 2% stop loss
                "take_profit": 0.04  # 4% take profit
            }
        } 