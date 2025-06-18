#!/bin/bash
# Setup script for 24/7 trading bot systemd service

echo "Setting up 24/7 Forex Trading Bot as systemd service..."

# Copy service file to systemd directory
sudo cp forex_trading_bot.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable forex_trading_bot.service

echo "Service installed!"
echo ""
echo "Available commands:"
echo "  Start service:   sudo systemctl start forex_trading_bot"
echo "  Stop service:    sudo systemctl stop forex_trading_bot"
echo "  Check status:    sudo systemctl status forex_trading_bot"
echo "  View logs:       sudo journalctl -u forex_trading_bot -f"
echo "  Disable autostart: sudo systemctl disable forex_trading_bot"
echo ""
echo "The bot will automatically restart if it crashes and start on system boot."