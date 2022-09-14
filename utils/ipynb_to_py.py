'''
Code adapted from StackOverflow answer on converting .ipynb to .py
'''

import sys,json

f = open('../new_experiments.ipynb', 'r') #input.ipynb
j = json.load(f)
of = open('../experiment_instance.py', 'w') #output.py
if j["nbformat"] >=4:
    for i,cell in enumerate(j["cells"]):
            for line in cell["source"]:
                    of.write(line)
            of.write('\n\n')
else:
    for i,cell in enumerate(j["worksheets"][0]["cells"]):
            for line in cell["input"]:
                    of.write(line)
            of.write('\n\n')

of.close()