from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
import random

languages = ['node', 'react', 'c++', 'HTML', 'python']
print(random.choice(foo))

from random import randrange
print(randrange(10))

tech_grammar = """
  S -> NP VP | VP NP
  NP -> Det N
  PP -> P NP
  VP -> 'Design' NP  | 'walked' PP 
  Det -> 'the' | 'a'
  N -> 'man' | 'park' | 'dog' | 'React' 
"""
grammar = CFG.fromstring(tech_grammar)
print(grammar)
for sentence in generate(grammar, n=39):
     print(' '.join(sentence))