@echo off
REM One-click daily run: refresh data, capture lines, price tonight's props.
REM
REM Safe to run repeatedly. Lineups post 2-3 hours before first pitch, so
REM running again later picks up more games. Results land in predictions\.
cd /d "%~dp0"
set PY=python

echo [1/5] Refreshing active rosters (drops IL and optioned players)...
%PY% active_rosters.py

echo.
echo [2/5] Refreshing Statcast through yesterday...
%PY% fetch_statcast.py 2026 --refresh-last

echo.
echo [3/5] Rebuilding the plate appearance table...
%PY% build_pa_events.py

echo.
echo [4/5] Capturing current market lines...
%PY% record_odds.py --days 1

echo.
echo [5/5] Pricing games with posted lineups...
%PY% live_props.py --sims 20000

echo.
echo Done. Open the newest file in the predictions folder.
pause
