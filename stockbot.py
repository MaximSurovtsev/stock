import telebot 
import net 
from yahooquery import Ticker

token = '1435238517:AAHBic3tl5gZYyuhisbRMYQFsbTsFkgU48M'

bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start'])
def hello(msg):
	bot.send_message(msg.chat.id, f'Привет, {msg.chat.username}! Моя задача помочь тебе в принятии решений на бирже. Отправь мне /stock ИМЯ_ТИКЕТА, а я отправлю цену закрытия следующей пятиминутной свечки')


@bot.message_handler(commands=['stock'])
def train(msg):
	comm, ticket = msg.text.split()
	ticker = Ticker(ticket, asynchronous=True)
	df = ticker.history(period='5d', interval='1m')
	if df[ticket] == 'No data found, symbol may be delisted':
		bot.send_message(msg.chat.id, 'Не нашел такой :(')
	else:
		bot.send_message(msg.chat.id, 'Нужно немного подождать')
		net.train(ticket)
		bot.send_message(msg.chat.id, f'Модель по {ticket} обучена и будет доступна в течение часа.')
		price = net.magic(ticket)
		bot.send_message(msg.chat.id, f'Предполагаемая цена закрытия следующей свечи: {price}')

bot.polling(none_stop=True)	