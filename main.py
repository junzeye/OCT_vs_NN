import json

with open('dic.out', 'r') as f:
    dic = json.load(f)
    print(dic)