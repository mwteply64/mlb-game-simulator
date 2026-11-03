@echo off
setlocal
REM Daily fantasy build. Refresh first, then price, then optimize.
REM
REM Steps 1-3 are the same refresh the props runner does, and they are not
REM optional: salaries and lineups are tonight's, so projecting them from a
REM stat file that ends days ago silently mixes fresh market data with stale
REM baseball. On 2026-08-22 the PA table ended 8/18 -- four days of games
REM missing against a slate posting that evening.
REM
REM Safe to run repeatedly. Lineups post 2-3 hours before first pitch, so a
REM later run sees more confirmed starters and fewer unknowns.
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
echo [4/5] Slates available tonight...
%PY% yahoo_dfs.py --slates

echo.
REM Field size drives the projected finish position, and only you know which
REM contest you are entering. Blank skips it: the lineups still build, they
REM just report ceilings without a finish estimate.
set CASH=24
set /p CASH=Entries in the SMALL contest? [24, 0 to skip]:

set GPP=360
set /p GPP=Entries in the LARGE contest? [360, 0 to skip]:

set LINEUPS=5
set /p LINEUPS=How many lineups per group? [5]:

echo.
echo [5/5] Building %LINEUPS% lineups per group...
echo        small field: %CASH% entries, ranked by mean projected points
echo        large field: %GPP% entries, ranked by ceiling
%PY% yahoo_dfs.py --auto --lineups %LINEUPS% --cash-field %CASH% --gpp-field %GPP%

echo.
echo Done. Results saved to predictions\dfs_^<date^>_^<time^>.*
echo   .txt         lineups grouped by contest type, ranked within group,
echo                with mean, sd, ceiling and projected finish
echo   _lineups.csv the lineups as data
echo   _pool.csv    every priced player, with vs_own_avg and own_start_avg
echo.
echo Salaries, lineups and market totals were pulled live at build time.
echo Roster rules enforced: max 6 per club, at least 3 clubs, no pitcher
echo against your own hitters, max 1 shared player between lineups.
pause
endlocal
