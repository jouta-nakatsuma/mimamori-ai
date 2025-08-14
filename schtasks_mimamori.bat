:: 毎晩22:00に nightly.ps1 を実行
schtasks /Create /TN "Mimamori_Nightly_Report" /SC DAILY /ST 22:00 /RU "SYSTEM" ^
  /TR "C:\Users\sunpi\ai\mimamori-ai\nightly.ps1" /RL HIGHEST /F
schtasks /Run /TN "Mimamori_Nightly_Report"

:: ログオン時に start_guardian.bat を実行
schtasks /Create /TN "Mimamori_Guardian_Startup" /SC ONLOGON /RU "SYSTEM" ^
  /TR "C:\Users\sunpi\ai\mimamori-ai\start_guardian.bat" /RL HIGHEST /F
