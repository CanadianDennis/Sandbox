from ofxparse import OfxParser, OfxPrinter
import csv

def main():
    
    csvdata = []
    
    with open('statement.csv') as csvfile:
        fieldnames = ['item_no', 'card_no', 'trans_date', 'post_date', 'amount', 'description']
        csvreader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for csvtrans in csvreader:
            csvdata.append(csvtrans)
    
    del(csvdata[0]) # Effective statement date
    del(csvdata[0]) # Column headings
    
    i = 0
    
    with open('statement.ofx') as infile:
        with open('output.ofx', 'w') as outfile:
            for ofxline in infile:
                if ofxline[:10] == '<DTPOSTED>':
                    ofxline = '<DTPOSTED>' + csvdata[i]['trans_date'] + ofxline[18:]
                    i = i + 1
                elif ofxline[:8] == '<TRNAMT>':
                    csvamount = -1.0 * float(csvdata[i-1]['amount'])
                    ofxamount = float(ofxline[8:])
                    if csvamount != ofxamount:
                        print('Warning: Transaction #%d amounts do not match:'% (i))
                        print(csvamount)
                        print(ofxamount)
                elif ofxline[:6] == '<NAME>':
                    csvpayee = csvdata[i-1]['description']
                    ofxpayee = ofxline[6:-1]
                    if csvpayee[:len(ofxpayee)] != ofxpayee:
                        print('Warning: Transaction #%d payees do not match:'% (i))
                        print(csvpayee)
                        print(ofxpayee)
                outfile.write(ofxline)
                
if __name__ == '__main__':
    main()
    