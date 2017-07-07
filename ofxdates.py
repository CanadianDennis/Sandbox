import sys
import csv
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QScrollArea
from PyQt5 import QtCore

def replace_dates(ofxpath, csvpath):
    
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


class MainWindow(QMainWindow):
  
    def __init__(self):
        super().__init__()

        self.setFixedSize(500,300)
        self.setWindowTitle('OFX Date Fix')
        
        self.setAcceptDrops(True)

        self.label = QLabel(self)
        self.label.setStyleSheet("""
                font: 10pt Courier;
                background-color: white;
            """)
        self.label.setText('Drag & drop OFX file here')
        self.label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)                
        
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.verticalScrollBar().rangeChanged.connect(self.resizeScroll)
        self.scrollArea.setWidget(self.label)
        
        self.setCentralWidget(self.scrollArea)

        self.show()


    def dragEnterEvent(self, e):
        
        e.accept()
        
        if e.mimeData().hasUrls:
            urls = e.mimeData().urls()
            if len(urls) == 1:
                filepath = str(urls[0].toLocalFile())
                if filepath[-4:] == '.ofx':
                    self.appendText('Drop!')
                else:
                    self.appendText('Not a OFX/CSV pair')
            else:
                self.appendText('Not a pair of files')
        else:
            self.appendText('Not a file')
        
    
    def dragLeaveEvent(self, e):
        
        self.appendText('Drag & drop OFX file here')


    def dropEvent(self, e):
        
        e.accept
        
        if e.mimeData().hasUrls:
            urls = e.mimeData().urls()
            if len(urls) == 1:
                filepath = str(urls[0].toLocalFile())
                if filepath[-4:] == '.ofx':
                    replace_dates(filepath)
    
    
    def appendText(self, text):
        self.label.setText(self.label.text() + '\n' + text)
        
    
    def resizeScroll(self, mini, maxi):
        self.scrollArea.verticalScrollBar().setValue(maxi)


if __name__ == '__main__':
  
    app = False
    app = QApplication(sys.argv)
    
    win = MainWindow()
    
    app.exec()
    