from __future__ import print_function

import os
import discord
from dotenv import load_dotenv
from chatbot import bot

import numpy as np
import tensorflow as tf

import argparse
import pickle
import copy
import sys
import html

from utils import TextLoader
from model import Model

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

client = discord.Client()

chatbot = bot()
args = chatbot.main()


def discordbot(net, sess, chars, vocab, max_length, beam_width, relevance, temperature, topn, Input_Text):
    states = chatbot.initial_state_with_relevance_masking(net, sess, relevance)
    #while True:
    user_input = Input_Text
    user_command_entered, reset, states, relevance, temperature, topn, beam_width = chatbot.process_user_command(
        user_input, states, relevance, temperature, topn, beam_width)
    if reset: states = chatbot.initial_state_with_relevance_masking(net, sess, relevance)
    if not user_command_entered:
        states = chatbot.forward_text(net, sess, states, relevance, vocab, chatbot.sanitize_text(vocab, "> " + user_input + "\n>"))
        computer_response_generator = chatbot.beam_search_generator(sess=sess, net=net,
            initial_state=copy.deepcopy(states), initial_sample=vocab[' '],
            early_term_token=vocab['\n'], beam_width=beam_width, forward_model_fn=chatbot.forward_with_mask,
            forward_args={'relevance':relevance, 'mask_reset_token':vocab['\n'], 'forbidden_token':vocab['>'],
                            'temperature':temperature, 'topn':topn})
        out_chars = []
        output = []
        for i, char_token in enumerate(computer_response_generator):
            out_chars.append(chars[char_token])
            output.append(chatbot.possibly_escaped_char(out_chars))
            #print(output, end='', flush=False)
            states = chatbot.forward_text(net, sess, states, relevance, vocab, chars[char_token])
            if i >= max_length: break
        #print(output)
        states = chatbot.forward_text(net, sess, states, relevance, vocab, chatbot.sanitize_text(vocab, "\n> "))
        return output

net, sess, chars, vocab, n, beam_width, relevance, temperature, topn = chatbot.discord_main(args)

@client.event
async def on_ready():
    
    print(f'{client.user.name} has connected to Discord!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    print(f'User: {message.content}')

    output = discordbot(net, sess, chars, vocab, n, beam_width, relevance, temperature, topn, message.content)

    response = ''.join(output)

    print(f'Bot: {response}')

    await message.channel.send(response)

client.run(TOKEN)