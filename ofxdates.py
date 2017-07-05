from ofxparse import OfxParser
import csv

with open('statement.ofx') as ofxfile:
    ofxdata = OfxParser.parse(ofxfile)

csvdata = []

with open('statement.csv') as csvfile:
    fieldnames = ['item_no', 'card_no', 'trans_date', 'post_date', 'amount', 'description']
    csvreader = csv.DictReader(csvfile, fieldnames=fieldnames)
    for csvtrans in csvreader:
        csvdata.append(csvtrans)

del(csvdata[0]) # Effective statement date
del(csvdata[0]) # Column headings

trans_count = len(csvdata)

for i in range(trans_count):
    csvtrans = csvdata[i]
    ofxtrans = ofxdata.account.statement.transactions[i]
    
    csvamount = -1.0 * float(csvtrans['amount'])
    ofxamount = float(ofxtrans.amount)
    
    if csvamount != ofxamount:
        print(csvamount, ofxamount)
        