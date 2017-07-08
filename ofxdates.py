import sys
import csv
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QScrollArea
from PyQt5 import QtCore

def replace_dates(ofxpath, csvpath):
    
    output = '----- Begin Processing -----\n'
    
    outpath = ofxpath[:-4] + '_fixed.ofx'
    
    csvdata = []
    with open(csvpath) as csvfile:
        fieldnames = ['item_no', 'card_no', 'trans_date', 'post_date', 'amount', 'description']
        csvreader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for csvtrans in csvreader:
            csvdata.append(csvtrans)
    
    del(csvdata[0]) # Effective statement date
    del(csvdata[0]) # Column headings
    
    i = 0
    with open(ofxpath) as infile:
        with open(outpath, 'w') as outfile:
            for ofxline in infile:
                if ofxline[:10] == '<DTPOSTED>':
                    ofxline = '<DTPOSTED>' + csvdata[i]['trans_date'] + ofxline[18:]
                    i = i + 1
                elif ofxline[:8] == '<TRNAMT>':
                    ofxamount = float(ofxline[8:])
                    csvamount = -1.0 * float(csvdata[i-1]['amount'])
                    if ofxamount != csvamount:
                        output = output + 'Warning: Transaction #' + str(i) + ' amounts do not match:\n'
                        output = output + 'OFX: ' + ofxamount + '\n'
                        output = output + 'CSV: ' + csvamount + '\n'
                elif ofxline[:6] == '<NAME>':
                    ofxpayee = ofxline[6:-1]
                    ofxpayee = ofxpayee.replace('&amp;', '&')
                    csvpayee = csvdata[i-1]['description']
                    if ofxpayee != csvpayee[:len(ofxpayee)]:
                        output = output + 'Warning: Transaction #' + str(i) + ' payees do not match:\n'
                        output = output + 'OFX: ' + ofxpayee + '\n'
                        output = output + 'CSV: ' + csvpayee + '\n'
                outfile.write(ofxline)
    
    output = output + 'Output: ' + outpath + '\n'
    output = output + '-----  End Processing  -----'
    
    return output


class MainWindow(QMainWindow):
  
    def __init__(self):
        super().__init__()
        
        self.setGeometry(100, 100, 500,300)
        self.setFixedSize(500,300)
        self.setWindowTitle('OFX Date Fix')
        
        self.setAcceptDrops(True)

        self.label = QLabel(self)
        self.label.setStyleSheet("""
                font: 10pt Courier;
                background-color: white;
            """)
        self.label.setText('Drag & drop OFX/CSV pair here')
        self.label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)                
        
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.verticalScrollBar().rangeChanged.connect(self.resizeScroll)
        self.scrollArea.setWidget(self.label)
        
        self.setCentralWidget(self.scrollArea)

        self.show()


    def dragEnterEvent(self, e):
        
        e.accept()
        
        urls = []
        
        if e.mimeData().hasUrls:
            for url in e.mimeData().urls():
                urls.append(str(url.toLocalFile()))
            if len(urls) == 2:
                if urls[0][-4:] == '.ofx' and urls[1][-4:] == '.csv':
                    self.appendText('Drop!')
                elif urls[0][-4:] == '.csv' and urls[1][-4:] == '.ofx':
                    self.appendText('Drop!')
                else:
                    self.appendText('Not a OFX/CSV pair')
            else:
                self.appendText('Not a pair of files')
        else:
            self.appendText('Not a file')
        
    
    def dragLeaveEvent(self, e):
        
        self.appendText('Drag & drop OFX/CSV pair here')


    def dropEvent(self, e):
        
        e.accept
        
        urls = []
            
        if e.mimeData().hasUrls:
            for url in e.mimeData().urls():
                urls.append(str(url.toLocalFile()))
            if len(urls) == 2:
                if urls[0][-4:] == '.ofx' and urls[1][-4:] == '.csv':
                    output = replace_dates(urls[0], urls[1])
                    self.appendText(output)
                elif urls[0][-4:] == '.csv' and urls[1][-4:] == '.ofx':
                    output = replace_dates(urls[1], urls[0])
                    self.appendText(output)


    def appendText(self, text):
        self.label.setText(self.label.text() + '\n' + text)


    def resizeScroll(self, mini, maxi):
        self.scrollArea.verticalScrollBar().setValue(maxi)


if __name__ == '__main__':
  
    app = False
    app = QApplication(sys.argv)
    
    win = MainWindow()
    
    app.exec()
    