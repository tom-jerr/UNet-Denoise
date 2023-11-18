import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox, QFileDialog
from denoise import *
import os
class MyWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.select_img);
        self.pushButton_2.clicked.connect(self.denoise_img);

    # 选择图片
    def select_img(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if imgName=='':
            QMessageBox.warning(self, "警告", "您没有选择任何文件！", QMessageBox.Yes)
        else:
            filename = os.path.basename(imgName)
            self.textEdit.setText(filename)
            # self.textBrowser.setText(imgType)
    def denoise_img(self):
        QMessageBox.information(self, "提示", "开始去噪！", QMessageBox.Yes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
