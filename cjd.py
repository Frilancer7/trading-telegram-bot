import requests

# Вставь сюда токен своего бота, который дал BotFather
TOKEN = "8226464157:AAHEWDYFP3EodFyaKIb-YRyN0MbojrQVPBI"

# Получаем все сообщения бота
r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates")
data = r.json()

# Выводим chat_id последнего сообщения
if data["result"]:
    chat_id = data["result"][-1]["message"]["chat"]["id"]
    print("Твой CHAT_ID:", chat_id)
else:
    print("Бот ещё не получил сообщений. Отправь сообщение боту в Telegram!")
