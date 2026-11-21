# Mech Royale bot (mechfight/1). Standard library only.
FROM python:3.13-slim
WORKDIR /bot
COPY bot.py navmap.py mechfight.py /bot/
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
