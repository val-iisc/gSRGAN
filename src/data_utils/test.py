import json

with open('../iNat19/train2019.json', 'r') as f:
    data = json.load(f)

print(data.keys())