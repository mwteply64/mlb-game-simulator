@echo off
REM Snapshot MLB betting lines. Scheduled hourly; each run appends a timestamped
REM snapshot so the closing line can be reconstructed as the last one before
REM first pitch. Safe to run at any hour -- duplicate snapshots are de-duped.
cd /d "%~dp0"
python record_odds.py --days 2 >> "logs\record_odds_log.txt" 2>&1
