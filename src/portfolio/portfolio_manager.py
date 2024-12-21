"""
Portfolio Management System
Handles portfolio tracking, performance analysis, and rebalancing
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging


class PortfolioManager:
    """
    Portfolio management system
    Tracks positions, analyzes performance, and manages rebalancing
    """

    def __init__(self, config: Dict):
        """
        Initialize portfolio manager with configuration
        Args:
            config: Portfolio management parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.positions = {}
        self.trades = []
        self.performance_history = []
        self.rebalance_thresholds = config.get('rebalance_thresholds', {
            'time': 7,  # Days between rebalances
            'deviation': 0.1,  # 10% deviation trigger
            'min_trade_size': 0.01  # 1% minimum trade size
        })

    async def get_portfolio_status(self, user_id: int) -> Dict:
        """
        Get current portfolio status and metrics
        Args:
            user_id: User identifier
        Returns:
            Dictionary containing portfolio status and metrics
        """
        try:
            # Get current positions and values
            positions = await self._get_current_positions(user_id)
            total_value = await self._calculate_total_value(positions)

            # Calculate performance metrics
            daily_change = await self._calculate_daily_change()
            performance = await self._calculate_performance_metrics()

            return {
                'timestamp': datetime.now().isoformat(),
                'total_value': total_value,
                'daily_change': daily_change,
                'assets': [
                    {
                        'symbol': symbol,
                        'amount': pos['amount'],
                        'value': await self._get_asset_value(symbol, pos['amount']),
                        'pnl': await self._calculate_position_pnl(symbol, pos),
                        'allocation': pos['amount'] * pos['current_price'] / total_value
                    }
                    for symbol, pos in positions.items()
                ],
                'metrics': performance
            }

        except Exception as e:
            self.logger.error(f"Portfolio status calculation failed: {str(e)}")
            raise

    async def update_position(self, trade: Dict):
        """
        Update portfolio positions after trade
        Args:
            trade: Executed trade details
        """
        try:
            asset = trade['asset']

            if asset not in self.positions:
                self.positions[asset] = {
                    'amount': 0,
                    'cost_basis': 0,
                    'trades': []
                }

            position = self.positions[asset]

            # Update position
            if trade['action'] == 'BUY':
                position['amount'] += trade['amount']
                position['cost_basis'] = (
                                                 (position['cost_basis'] * position['amount']) +
                                                 (trade['price'] * trade['amount'])
                                         ) / position['amount']
            else:
                position['amount'] -= trade['amount']

            # Record trade
            position['trades'].append({
                'timestamp': datetime.now().isoformat(),
                'type': trade['action'],
                'amount': trade['amount'],
                'price': trade['price']
            })

            # Update performance history
            await self._update_performance_history()

        except Exception as e:
            self.logger.error(f"Position update failed: {str(e)}")
            raise

    async def check_rebalance_needed(self) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            # Check time since last rebalance
            last_rebalance = self._get_last_rebalance_time()
            if (datetime.now() - last_rebalance).days < self.rebalance_thresholds['time']:
                return False

            # Check allocation deviations
            target_allocations = self.config['target_allocations']
            current_allocations = await self._get_current_allocations()

            for asset, target in target_allocations.items():
                current = current_allocations.get(asset, 0)
                if abs(current - target) > self.rebalance_thresholds['deviation']:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Rebalance check failed: {str(e)}")
            return False

    async def generate_rebalance_trades(self) -> List[Dict]:
        """
        Generate trades needed for rebalancing
        Returns:
            List of trades required for rebalancing
        """
        try:
            trades = []
            total_value = await self._calculate_total_value(self.positions)
            current_allocations = await self._get_current_allocations()

            for asset, target in self.config['target_allocations'].items():
                current = current_allocations.get(asset, 0)
                difference = target - current

                # Check if deviation exceeds threshold
                if abs(difference) > self.rebalance_thresholds['deviation']:
                    trade_value = difference * total_value

                    # Check minimum trade size
                    if abs(trade_value) >= (total_value * self.rebalance_thresholds['min_trade_size']):
                        price = await self._get_market_price(asset)
                        amount = abs(trade_value / price)

                        trades.append({
                            'asset': asset,
                            'action': 'BUY' if difference > 0 else 'SELL',
                            'amount': amount,
                            'expected_price': price,
                            'reason': 'rebalance'
                        })

            return trades

        except Exception as e:
            self.logger.error(f"Rebalance trade generation failed: {str(e)}")
            return []

    async def calculate_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics
        Returns:
            Dictionary containing various performance metrics
        """
        try:
            history = pd.DataFrame(self.performance_history)
            returns = history['total_value'].pct_change()

            metrics = {
                'total_return': self._calculate_total_return(),
                'daily_returns': {
                    'mean': returns.mean(),
                    'std': returns.std(),
                    'skew': returns.skew(),
                    'kurtosis': returns.kurtosis()
                },
                'risk_metrics': {
                    'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                    'sortino_ratio': self._calculate_sortino_ratio(returns),
                    'max_drawdown': self._calculate_max_drawdown(history),
                    'var_95': self._calculate_var(returns, 0.95),
                    'cvar_95': self._calculate_cvar(returns, 0.95)
                },
                'trading_metrics': {
                    'win_rate': self._calculate_win_rate(),
                    'profit_factor': self._calculate_profit_factor(),
                    'avg_win_loss': self._calculate_avg_win_loss()
                }
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {str(e)}")
            return {}

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return (returns.mean() - self.config['risk_free_rate']) / returns.std() * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0
        downside_std = negative_returns.std()
        if downside_std == 0:
            return 0
        return (returns.mean() - self.config['risk_free_rate']) / downside_std * np.sqrt(252)

    def _calculate_max_drawdown(self, history: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        rolling_max = history['total_value'].expanding().max()
        drawdowns = history['total_value'] / rolling_max - 1
        return drawdowns.min()

    async def generate_report(self, period: str = 'daily') -> Dict:
        """
        Generate detailed portfolio report
        Args:
            period: Report period ('daily', 'weekly', 'monthly')
        Returns:
            Dictionary containing portfolio report
        """
        try:
            # Get current portfolio status
            status = await self.get_portfolio_status(user_id=None)

            # Calculate period performance
            period_performance = await self._calculate_period_performance(period)

            # Get trade history analysis
            trade_analysis = self._analyze_trades(period)

            # Generate allocation analysis
            allocation_analysis = await self._analyze_allocations()

            report = {
                'timestamp': datetime.now().isoformat(),
                'period': period,
                'summary': {
                    'total_value': status['total_value'],
                    'period_return': period_performance['return'],
                    'period_pnl': period_performance['pnl']
                },
                'performance': {
                    'returns': period_performance,
                    'metrics': await self.calculate_metrics()
                },
                'trading': {
                    'total_trades': len(trade_analysis['trades']),
                    'win_rate': trade_analysis['win_rate'],
                    'profit_factor': trade_analysis['profit_factor'],
                    'largest_trades': trade_analysis['largest_trades']
                },
                'allocations': {
                    'current': allocation_analysis['current'],
                    'target': allocation_analysis['target'],
                    'deviations': allocation_analysis['deviations']
                },
                'risk_analysis': {
                    'var': status['metrics']['risk_metrics']['var_95'],
                    'cvar': status['metrics']['risk_metrics']['cvar_95'],
                    'concentration_risk': self._calculate_concentration_risk(),
                    'correlation_matrix': await self._calculate_correlation_matrix()
                }
            }

            return report

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {}

    async def optimize_portfolio(self, constraints: Dict = None) -> Dict:
        """
        Optimize portfolio allocations
        Args:
            constraints: Optional optimization constraints
        Returns:
            Dictionary containing optimal allocations
        """
        try:
            # Get historical data
            history = self._get_historical_data()

            # Calculate expected returns and covariance
            returns = self._calculate_expected_returns(history)
            cov_matrix = self._calculate_covariance_matrix(history)

            # Apply optimization algorithm (e.g., Modern Portfolio Theory)
            optimal_weights = self._optimize_weights(
                returns,
                cov_matrix,
                constraints or self.config.get('optimization_constraints', {})
            )

            return {
                'weights': optimal_weights,
                'metrics': self._calculate_portfolio_metrics(optimal_weights, returns, cov_matrix),
                'comparison': self._compare_with_current(optimal_weights)
            }

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {str(e)}")
            return {}

    def _optimize_weights(self, returns: pd.Series, cov_matrix: pd.DataFrame,
                         constraints: Dict) -> Dict:
        """Optimize portfolio weights using efficient frontier"""
        try:
            # Implement portfolio optimization using scipy.optimize
            # This is a simplified version - actual implementation would be more complex
            from scipy.optimize import minimize

            def objective(weights):
                portfolio_return = np.sum(returns * weights)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                # Maximize Sharpe Ratio
                return -(portfolio_return - self.config['risk_free_rate']) / portfolio_risk

            # Add constraints and bounds based on input parameters
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
            ]

            bounds = [(0, constraints.get('max_weight', 1)) for _ in range(len(returns))]

            result = minimize(objective,
                            x0=np.array([1/len(returns)] * len(returns)),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints_list)

            if result.success:
                return dict(zip(returns.index, result.x))
            else:
                raise Exception("Optimization failed to converge")

        except Exception as e:
            self.logger.error(f"Weight optimization failed: {str(e)}")
            return {}