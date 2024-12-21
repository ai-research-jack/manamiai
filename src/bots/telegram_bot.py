"""
Telegram Trading Bot Implementation
Provides automated trading functionality through Telegram interface
"""

from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    Filters
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import Optional, Dict, List
import asyncio
import logging
from datetime import datetime


class TradingTelegramBot:
    """
    Telegram bot for automated trading
    Handles user commands, executes trades, and provides market updates
    """

    def __init__(self, config: Dict):
        """
        Initialize the trading bot
        Args:
            config: Bot configuration including API keys and trading parameters
        """
        self.config = config
        self.updater = None
        self.bot = None
        self.active_trades = {}
        self.user_sessions = {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.trend_analyzer = MarketTrendAnalyzer(config)

    async def initialize(self):
        """Initialize bot and connect to required services"""
        try:
            # Set up Telegram bot
            self.updater = Updater(self.config['telegram_token'])
            self.bot = self.updater.bot

            # Register command handlers
            dp = self.updater.dispatcher
            dp.add_handler(CommandHandler("start", self.handle_start))
            dp.add_handler(CommandHandler("trade", self.handle_trade))
            dp.add_handler(CommandHandler("status", self.handle_status))
            dp.add_handler(CommandHandler("portfolio", self.handle_portfolio))
            dp.add_handler(CommandHandler("alerts", self.handle_alerts))
            dp.add_handler(CallbackQueryHandler(self.handle_callback))

            # Start the bot
            self.updater.start_polling()
            self.logger.info("Telegram bot successfully initialized")

            # Start background tasks
            asyncio.create_task(self.market_monitor())
            asyncio.create_task(self.portfolio_monitor())

        except Exception as e:
            self.logger.error(f"Bot initialization failed: {str(e)}")
            raise

    async def handle_start(self, update, context):
        """
        Handle /start command
        Introduces the bot and shows available commands
        """
        welcome_text = (
            "🤖 Welcome to ManaMi Trading Bot!\n\n"
            "Available commands:\n"
            "/trade - Execute trades\n"
            "/status - Check market status\n"
            "/portfolio - View portfolio\n"
            "/alerts - Configure alerts"
        )

        keyboard = [
            [InlineKeyboardButton("Trade", callback_data='trade'),
             InlineKeyboardButton("Status", callback_data='status')],
            [InlineKeyboardButton("Portfolio", callback_data='portfolio'),
             InlineKeyboardButton("Alerts", callback_data='alerts')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(welcome_text, reply_markup=reply_markup)

    async def handle_trade(self, update, context):
        """
        Handle /trade command
        Process trading commands and execute trades
        """
        try:
            user_id = update.effective_user.id
            if not self._is_user_authorized(user_id):
                await update.message.reply_text("⚠️ Unauthorized user")
                return

            command_parts = context.args
            if len(command_parts) < 3:
                await update.message.reply_text(
                    "📝 Usage: /trade <action> <asset> <amount>\n"
                    "Example: /trade buy BTC 0.1"
                )
                return

            action, asset, amount = command_parts

            # Validate trade parameters
            if not await self._validate_trade(action, asset, amount):
                await update.message.reply_text("❌ Invalid trade parameters")
                return

            # Execute trade
            trade_result = await self.execute_trade({
                'user_id': user_id,
                'action': action.upper(),
                'asset': asset.upper(),
                'amount': float(amount),
                'timestamp': datetime.now().isoformat()
            })

            if trade_result['success']:
                await update.message.reply_text(
                    f"✅ Trade executed successfully!\n"
                    f"Details: {trade_result['details']}"
                )
            else:
                await update.message.reply_text(
                    f"❌ Trade failed: {trade_result['error']}"
                )

        except Exception as e:
            self.logger.error(f"Trade handling failed: {str(e)}")
            await update.message.reply_text("❌ An error occurred while processing your trade")

    async def execute_trade(self, trade_params: Dict) -> Dict:
        """
        Execute a trade with given parameters
        Args:
            trade_params: Dictionary containing trade details
        Returns:
            Dictionary containing trade execution results
        """
        try:
            # Risk check
            risk_check = await self.risk_manager.check_trade_risk(trade_params)
            if not risk_check['allowed']:
                return {
                    'success': False,
                    'error': f"Risk check failed: {risk_check['reason']}"
                }

            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(trade_params)

            # Market check
            market_check = await self._check_market_conditions(trade_params['asset'])
            if not market_check['favorable']:
                return {
                    'success': False,
                    'error': f"Unfavorable market conditions: {market_check['reason']}"
                }

            # Execute order
            order_result = await self._place_order({
                **trade_params,
                'position_size': position_size
            })

            if order_result['success']:
                # Update portfolio
                await self.portfolio_manager.update_position(order_result['order'])

                return {
                    'success': True,
                    'details': {
                        'order_id': order_result['order']['id'],
                        'executed_price': order_result['order']['price'],
                        'size': position_size,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            else:
                return {
                    'success': False,
                    'error': order_result['error']
                }

        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            return {
                'success': False,
                'error': "Internal execution error"
            }

    async def market_monitor(self):
        """
        Background task to monitor market conditions
        Generates alerts and signals based on market analysis
        """
        while True:
            try:
                for asset in self.config['monitored_assets']:
                    # Analyze market conditions
                    analysis = await self.trend_analyzer.analyze_market_sentiment({
                        'asset': asset,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Check for significant changes
                    if self._is_significant_change(analysis):
                        await self._send_alerts(asset, analysis)

                    # Generate trading signals
                    signals = await self.trend_analyzer.generate_trading_signals()
                    for signal in signals:
                        if signal['confidence'] >= self.config['signal_threshold']:
                            await self._process_trading_signal(signal)

                # Sleep interval
                await asyncio.sleep(self.config['market_check_interval'])

            except Exception as e:
                self.logger.error(f"Market monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Error backoff

    async def handle_portfolio(self, update, context):
        """
        Handle /portfolio command
        Display current portfolio status and performance metrics
        """
        try:
            user_id = update.effective_user.id
            portfolio_data = await self.portfolio_manager.get_portfolio_status(user_id)

            # Format portfolio message
            message = "📊 Portfolio Status\n\n"
            message += f"Total Value: ${portfolio_data['total_value']:,.2f}\n"
            message += f"24h Change: {portfolio_data['daily_change']}%\n\n"

            # Add asset breakdown
            message += "Asset Breakdown:\n"
            for asset in portfolio_data['assets']:
                message += (f"• {asset['symbol']}: {asset['amount']} "
                            f"(${asset['value']:,.2f})\n")

            # Add performance metrics
            message += f"\nROI: {portfolio_data['roi']}%\n"
            message += f"Winning Trades: {portfolio_data['win_rate']}%"

            await update.message.reply_text(message)

        except Exception as e:
            self.logger.error(f"Portfolio handling error: {str(e)}")
            await update.message.reply_text("❌ Error retrieving portfolio data")

    async def handle_alerts(self, update, context):
        """
        Handle /alerts command
        Configure price and trend alerts
        """
        try:
            keyboard = [
                [InlineKeyboardButton("Price Alerts", callback_data='price_alert'),
                 InlineKeyboardButton("Trend Alerts", callback_data='trend_alert')],
                [InlineKeyboardButton("View Active Alerts", callback_data='view_alerts')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "⚙️ Alert Configuration\n\n"
                "Choose alert type to configure:",
                reply_markup=reply_markup
            )

        except Exception as e:
            self.logger.error(f"Alert handling error: {str(e)}")
            await update.message.reply_text("❌ Error configuring alerts")

    async def _validate_trade(self, action: str, asset: str, amount: str) -> bool:
        """
        Validate trade parameters
        Args:
            action: Trade action (buy/sell)
            asset: Asset symbol
            amount: Trade amount
        Returns:
            bool: Whether trade parameters are valid
        """
        try:
            # Validate action
            if action.lower() not in ['buy', 'sell']:
                return False

            # Validate asset
            if asset.upper() not in self.config['supported_assets']:
                return False

            # Validate amount
            amount = float(amount)
            if amount <= 0:
                return False

            return True

        except ValueError:
            return False

    def _is_significant_change(self, analysis: Dict) -> bool:
        """
        Check if market change is significant enough for alert
        Args:
            analysis: Market analysis results
        Returns:
            bool: Whether change is significant
        """
        thresholds = self.config['alert_thresholds']

        # Check price change
        if abs(analysis['price_change']) >= thresholds['price']:
            return True

        # Check sentiment change
        if abs(analysis['sentiment_change']) >= thresholds['sentiment']:
            return True

        # Check volume change
        if analysis['volume_change'] >= thresholds['volume']:
            return True

        return False

    async def _send_alerts(self, asset: str, analysis: Dict):
        """
        Send alerts to subscribed users
        Args:
            asset: Asset symbol
            analysis: Analysis results triggering the alert
        """
        alert_message = self._format_alert_message(asset, analysis)

        for user_id in self.user_sessions:
            if self._should_alert_user(user_id, asset, analysis):
                try:
                    await self.bot.send_message(
                        chat_id=user_id,
                        text=alert_message
                    )
                except Exception as e:
                    self.logger.error(f"Alert sending failed for user {user_id}: {str(e)}")

    def _format_alert_message(self, asset: str, analysis: Dict) -> str:
        """Format alert message based on analysis results"""
        message = f"🚨 {asset} Alert\n\n"

        if abs(analysis['price_change']) >= self.config['alert_thresholds']['price']:
            message += (f"Price {analysis['price_change']}% "
                        f"{'⬆️' if analysis['price_change'] > 0 else '⬇️'}\n")

        if abs(analysis['sentiment_change']) >= self.config['alert_thresholds']['sentiment']:
            message += f"Significant sentiment change detected\n"

        if analysis['volume_change'] >= self.config['alert_thresholds']['volume']:
            message += f"Unusual volume increase: {analysis['volume_change']}x\n"

        message += f"\nCurrent Price: ${analysis['current_price']:,.2f}"
        return message