import os
import discord
from dotenv import load_dotenv
import chatbot

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

client = discord.Client()

@client.event
async def on_ready():
    
    
    print(f'{client.user.name} has connected to Discord!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    args = chatbot.main()
    output = chatbot.discord_main(args, message.content)
    response = ''.join(output)

    await message.channel.send(response)

client.run(TOKEN)