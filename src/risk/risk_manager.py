"""
Risk Management System
Handles position sizing, risk limits, and portfolio protection
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging


class RiskManager:
    """
    Risk management system for trading operations
    Implements position sizing, risk limits, and drawdown protection
    """

    def __init__(self, config: Dict):
        """
        Initialize risk manager with configuration
        Args:
            config: Risk management parameters and limits
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.position_limits = config.get('position_limits', {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_asset_exposure': 0.2,  # 20% per asset
            'max_total_exposure': 0.8  # 80% total exposure
        })
        self.risk_limits = config.get('risk_limits', {
            'max_daily_drawdown': 0.02,  # 2% daily
            'max_total_drawdown': 0.1,  # 10% total
            'stop_loss': 0.05  # 5% stop loss
        })
        self.positions = {}
        self.daily_stats = self._initialize_daily_stats()

    async def check_trade_risk(self, trade: Dict) -> Dict:
        """
        Check if trade meets risk management criteria
        Args:
            trade: Proposed trade details
        Returns:
            Dictionary with risk check results
        """
        try:
            checks = {
                'position_size': self._check_position_size(trade),
                'exposure': self._check_exposure_limits(trade),
                'drawdown': await self._check_drawdown_limits(),
                'volatility': self._check_volatility(trade['asset'])
            }

            allowed = all(checks.values())
            reason = None if allowed else self._get_rejection_reason(checks)

            return {
                'allowed': allowed,
                'reason': reason,
                'checks': checks
            }

        except Exception as e:
            self.logger.error(f"Risk check failed: {str(e)}")
            return {
                'allowed': False,
                'reason': 'Risk check error',
                'checks': {}
            }

    async def calculate_position_size(self, trade_params: Dict) -> float:
        """
        Calculate optimal position size for trade
        Args:
            trade_params: Trade parameters including asset and direction
        Returns:
            Optimal position size as percentage of portfolio
        """
        try:
            # Get account and market data
            account_size = await self._get_account_size()
            volatility = await self._get_asset_volatility(trade_params['asset'])

            # Calculate base position size using Kelly Criterion
            win_rate = self._calculate_historical_win_rate(trade_params['asset'])
            reward_ratio = self._calculate_reward_ratio(trade_params)
            kelly = win_rate - ((1 - win_rate) / reward_ratio)

            # Apply risk adjustments
            adjusted_size = kelly * self._get_risk_adjustment_factor()

            # Apply limits
            max_size = min(
                self.position_limits['max_position_size'],
                self._get_asset_specific_limit(trade_params['asset'])
            )
            position_size = min(adjusted_size, max_size)

            # Convert to actual size
            final_size = position_size * account_size

            return final_size

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            return 0.0

    def _check_position_size(self, trade: Dict) -> bool:
        """Check if trade size meets position limits"""
        try:
            # Calculate relative position size
            account_size = self._get_account_value()
            relative_size = trade['amount'] / account_size

            # Check against limits
            if relative_size > self.position_limits['max_position_size']:
                self.logger.warning(f"Position size {relative_size:.2%} exceeds limit")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Position size check failed: {str(e)}")
            return False

    def _check_exposure_limits(self, trade: Dict) -> bool:
        """Check asset and total exposure limits"""
        try:
            asset = trade['asset']
            current_exposure = self._calculate_current_exposure()

            # Check asset-specific exposure
            asset_exposure = self.positions.get(asset, 0) + trade['amount']
            if asset_exposure > self.position_limits['max_asset_exposure']:
                return False

            # Check total exposure
            total_exposure = current_exposure + trade['amount']
            if total_exposure > self.position_limits['max_total_exposure']:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Exposure check failed: {str(e)}")
            return False

    async def _check_drawdown_limits(self) -> bool:
        """Check current drawdown against limits"""
        try:
            # Calculate current drawdowns
            daily_drawdown = await self._calculate_daily_drawdown()
            total_drawdown = await self._calculate_total_drawdown()

            # Check against limits
            if daily_drawdown > self.risk_limits['max_daily_drawdown']:
                self.logger.warning(f"Daily drawdown {daily_drawdown:.2%} exceeds limit")
                return False

            if total_drawdown > self.risk_limits['max_total_drawdown']:
                self.logger.warning(f"Total drawdown {total_drawdown:.2%} exceeds limit")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Drawdown check failed: {str(e)}")
            return False

    def adjust_stop_loss(self, position: Dict) -> Dict:
        """
        Dynamically adjust stop loss levels
        Args:
            position: Current position details
        Returns:
            Updated position with new stop loss
        """
        try:
            entry_price = position['entry_price']
            current_price = position['current_price']
            original_stop = position['stop_loss']

            # Calculate trailing stop
            if position['direction'] == 'BUY':
                trailing_stop = current_price * (1 - self.risk_limits['stop_loss'])
                new_stop = max(trailing_stop, original_stop)
            else:
                trailing_stop = current_price * (1 + self.risk_limits['stop_loss'])
                new_stop = min(trailing_stop, original_stop)

            return {
                **position,
                'stop_loss': new_stop,
                'stop_updated': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Stop loss adjustment failed: {str(e)}")
            return position

    def _calculate_reward_ratio(self, trade: Dict) -> float:
        """Calculate reward to risk ratio for trade"""
        try:
            entry = trade['entry_price']
            stop = trade['stop_loss']
            target = trade['take_profit']

            risk = abs(entry - stop)
            reward = abs(target - entry)

            return reward / risk if risk > 0 else 0

        except Exception as e:
            self.logger.error(f"Reward ratio calculation failed: {str(e)}")
            return 0.0

    def _get_risk_adjustment_factor(self) -> float:
        """Calculate risk adjustment factor based on market conditions"""
        try:
            factors = {
                'volatility': self._get_volatility_factor(),
                'correlation': self._get_correlation_factor(),
                'market_regime': self._get_market_regime_factor()
            }

            return np.mean(list(factors.values()))

        except Exception as e:
            self.logger.error(f"Risk adjustment calculation failed: {str(e)}")
            return 0.5  # Default to conservative adjustment

    async def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report
        Returns:
            Dictionary containing risk metrics and analysis
        """
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'exposure': {
                    'total': self._calculate_current_exposure(),
                    'by_asset': self.positions
                },
                'drawdown': {
                    'daily': await self._calculate_daily_drawdown(),
                    'total': await self._calculate_total_drawdown()
                },
                'risk_metrics': {
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'sortino_ratio': self._calculate_sortino_ratio(),
                    'value_at_risk': self._calculate_var()
                },
                'position_analysis': self._analyze_positions()
            }

        except Exception as e:
            self.logger.error(f"Risk report generation failed: {str(e)}")
            return {}