import json

f = open("text_lexicon.txt", "r")
word2phonelist = dict()
all_phone = list()
phone2idx = dict()
idx2phone = dict()

lines = f.readlines()

for line in lines:
    word, phones = line.split()
    phone_list = phones.split("-")
    tmp = list()
    for p in phone_list:
        p_norm = p.replace("CH_", "")
        if p_norm not in phone2idx:
            i = len(phone2idx)
            phone2idx[p_norm] = i+1
            idx2phone[i+1] = p_norm
        tmp.append(p_norm)

    if word not in word2phonelist:
        word2phonelist[word] = tmp

phone2idx["P"] = 0
idx2phone[0] = "P"
length = len(phone2idx)
phone2idx['oov'] = length
phone2idx['S'] = length+1
idx2phone[length] = 'oov'
idx2phone[length+1] = 'S'

json.dump(idx2phone, open("idx2phone.json","w"))
json.dump(phone2idx, open("phone2idx.json","w"))
json.dump(word2phonelist, open("word2phonelist.json","w"))
