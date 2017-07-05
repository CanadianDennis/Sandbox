from ofxparse import OfxParser, OfxPrinter
import csv

def main():

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
        
        csvamount = -1.0 * float(csvdata[i]['amount'])
        ofxamount = float(ofxdata.account.statement.transactions[i].amount)
        if csvamount != ofxamount:
            print( 'Error: Transaction #%d does not match'% (i+1) )
            return
        
        ofxprint = OfxPrinter(ofx=ofxdata, filename='output.ofx')
        ofxprint.write()
        
if __name__ == '__main__':
    main()
    