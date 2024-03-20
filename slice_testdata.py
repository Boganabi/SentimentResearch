from csv import DictWriter
import json

fieldnames=['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'verified_purchase', 'helpful_vote']

with open("Electronics.jsonl", 'r', encoding="utf-8") as inp, open("testdata.csv", 'w', encoding="utf-8", newline="") as outp:
    writer = DictWriter(outp, fieldnames=[
            'rating', 'title', 'text',
            'images', 'asin', 'parent_asin',
            'user_id', 'timestamp', 
            'verified_purchase', 'helpful_vote'])
    lineCount = 0
    for line in inp:
        row = json.loads(line)
        writer.writerow(row)
        lineCount += 1
        if lineCount > 3000: # to limit the amount of data we have
            break