@echo off
cd /d %~dp0..

python code\05_entrainement.py --config c5_L16

python code\05_entrainement.py --config c5_L8  --modeles transformer
python code\05_entrainement.py --config c5_L32 --modeles transformer
python code\05_entrainement.py --config c5_L48 --modeles transformer

python code\05_entrainement.py --config c1_L8  --modeles transformer
python code\05_entrainement.py --config c1_L16 --modeles transformer

echo.
echo ==================== TOUT EST TERMINE ====================
pause
