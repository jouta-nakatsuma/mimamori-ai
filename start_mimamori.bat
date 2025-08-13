schtasks /Create /TN "Mimamori_Manual" /SC ONCE /ST 00:00 /RU "SYSTEM" `
  /TR "C:\Users\sunpi\ai\mimamori-ai\start_mimamori.bat" /RL HIGHEST /F
schtasks /Run /TN "Mimamori_Manual"
