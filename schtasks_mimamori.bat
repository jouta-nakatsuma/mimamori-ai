schtasks /Create /TN "Mimamori_Nightly_Report" /SC DAILY /ST 22:00 /RU "SYSTEM" `
  /TR "C:\Users\sunpi\ai\mimamori-ai\nightly.ps1" /RL HIGHEST /F
schtasks /Run /TN "Mimamori_Nightly_Report"
