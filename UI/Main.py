from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import tensorflow as tf
from PyQt5.QtCore import Qt
from Drawingboard import App
from Court import Court
import WGAN

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        #self.graph, self.saver, self.config, self.data_factory = WGAN.Load_Model()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 720)
        self.play = App(MainWindow)
        self.court = Court(MainWindow)
        MainWindow.setAutoFillBackground(True)
        p = MainWindow.palette()
        p.setColor(MainWindow.backgroundRole(), Qt.white)
        MainWindow.setPalette(p)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(50, 50, 570, 580))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.graphicsView = QtWidgets.QGraphicsView(self.horizontalLayoutWidget)
        self.graphicsView.setObjectName("DrawingBoard")
        #Play
        self.horizontalLayout.addWidget(self.play)

        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(630, 50, 570, 580))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.horizontalLayoutWidget_2)
        self.graphicsView_2.setObjectName("Animation")
        #Animation
        self.horizontalLayout_2.addWidget(self.court)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.showMessage("BasketballGAN")

        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(QtWidgets.qApp.quit)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.Buttons(self.play)
        self.Radio_buttons()
        self.player()

        self.gen_widget = QtWidgets.QWidget(self.centralwidget)
        self.text_label = QtWidgets.QLabel("Generate",self.gen_widget)
        self.text_label.move(0, 50)
        self.gen_widget.setGeometry(QtCore.QRect(595, 300, 100, 100))

        self.modelButton = QtWidgets.QPushButton( self.gen_widget)
        self.modelButton.setGeometry(QtCore.QRect(8, 0, 50, 50))
        self.modelButton.setObjectName("GenButton")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap('./images/arrow.png'))
        self.modelButton.setIcon(icon)
        self.modelButton.setIconSize(QtCore.QSize(50,50))
        self.modelButton.clicked.connect(self.run_model)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BasketballGAN"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))


    def run_model(self):
        self.statusbar.showMessage("Saving...")
        self.play.scene.savePos()
        self.statusbar.showMessage("Generating...")
        WGAN.run_Model(self.graph, self.saver, self.config, self.data_factory)
        self.ani_button.setChecked(True)
        self.statusbar.showMessage("Done!")

        self.statusbar.showMessage("BasketballGAN")

    def Buttons(self,play):
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(240, 630, 200, 40))
        self.widget.setObjectName("ButtonBar")

        self.clearButton = QtWidgets.QPushButton("Clear",self.widget)
        self.clearButton.setGeometry(QtCore.QRect(10, 0, 75, 25))
        self.clearButton.setObjectName("clearButton")
        self.clearButton.clicked.connect(play.scene.clear_c)

        self.setButton = QtWidgets.QPushButton("Set",self.widget)
        self.setButton.setGeometry(QtCore.QRect(90, 0, 75, 25))
        self.setButton.setObjectName("setButton")
        self.setButton.clicked.connect(play.scene.setDraw)

        '''
        self.aniButton = QtWidgets.QPushButton("Save",self.widget)
        self.aniButton.setGeometry(QtCore.QRect(170, 0, 75, 23))
        self.aniButton.setObjectName("SaveButton")
        self.aniButton.clicked.connect(play.scene.savePos)
        '''
        return None

    def Radio_buttons(self):
        self.radio_widget = QtWidgets.QWidget(self.centralwidget)
        self.radio_widget.setGeometry(QtCore.QRect(660, 650, 870, 40))
        self.radio_widget.setObjectName("radio")

        self.ani_button = QRadioButton("   Sketch\nAnimation", self.radio_widget,)

        self.ani_button.setChecked(False)
        self.ani_button.toggled.connect(lambda: self.btnstate(self.ani_button))
        self.ani_button.move(150, 0)

        self.sim_button = QRadioButton("      Play\nSimulation", self.radio_widget)
        self.sim_button.setChecked(False)
        self.sim_button.toggled.connect(lambda: self.btnstate(self.sim_button))
        self.sim_button.move(300, 0)

        self.len_ = 0

    def btnstate(self, b):
        if b.text() == "   Sketch\nAnimation":
            if b.isChecked() == True:
                self.court.is_playing = True
                self.playButton.setIcon(self.player_widget.style().standardIcon(QStyle.SP_MediaPlay))
                self.playButton.setEnabled(True)

                self.court.loaded = True
                self.court.canvas.frame_id = 0
                self.positionSlider.setValue(0)
                self.court.canvas.on_start()
                self.court.sim_timer.stop()

                self.len_ = self.court.get_len(1)
                self.durationChanged(self.len_)
            else:
                print("Switch Animation")
        if b.text() == "      Play\nSimulation":
            if b.isChecked() == True:
                self.court.is_playing = True
                self.playButton.setIcon(self.player_widget.style().standardIcon(QStyle.SP_MediaPlay))
                self.playButton.setEnabled(True)

                self.court.loaded = True
                self.court.canvas.frame_id = 0
                self.positionSlider.setValue(0)
                self.court.ani_timer.stop()
                self.court.start_sim()

                self.len_ = self.court.get_len(2)
                self.durationChanged(self.len_)
            else:
                print("Switch Simulate")

    def player(self):
        self.player_widget = QtWidgets.QWidget(self.centralwidget)
        self.player_widget.setGeometry(QtCore.QRect(705, 625, 870, 40))
        self.player_widget.setObjectName("Player")

        self.playButton = QPushButton("", self.player_widget)
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.player_widget.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setFixedSize(50, 30)
        self.playButton.clicked.connect(self.play_)
        self.playButton.move(0, 0)

        self.positionSlider = QSlider(QtCore.Qt.Horizontal, self.player_widget)
        self.positionSlider.setRange(0, 1000)
        self.positionSlider.move(50, 7)

        self.positionSlider.setMinimumWidth(400)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.player_timer = QtCore.QTimer(self.centralwidget)
        self.player_timer.timeout.connect(self.positionChanged)

    def play_(self):
        if self.court.is_playing == True:
            self.playButton.setIcon(self.player_widget.style().standardIcon(QStyle.SP_MediaPause))
            self.court.is_playing = False
            if self.ani_button.isChecked() == True:
                self.court.ani_timer.start(200)
                self.player_timer.start(200)
            elif self.sim_button.isChecked() == True:
                self.court.sim_timer.start(200)
                self.player_timer.start(200)

        elif self.court.is_playing == False:
            self.playButton.setIcon(self.player_widget.style().standardIcon(QStyle.SP_MediaPlay))
            self.court.is_playing = True

            if self.ani_button.isChecked() == True:
                self.court.ani_timer.stop()
            elif self.sim_button.isChecked() == True:
                self.court.sim_timer.stop()


    def positionChanged(self):
        position = self.court.canvas.frame_id
        self.positionSlider.setValue(position)
        if position == self.len_-1:
            self.playButton.setIcon(self.player_widget.style().standardIcon(QStyle.SP_MediaPlay))
            self.court.is_playing = True
            if self.ani_button.isChecked() == True:
                self.court.ani_timer.stop()
            elif self.sim_button.isChecked() == True:
                self.court.sim_timer.stop()

    def durationChanged(self, duration):
        self.positionSlider.setRange(1, duration-1)

    def setPosition(self):
        self.player_timer.stop()
        pos = self.positionSlider.value()
        self.court.canvas.frame_id = pos
        if self.ani_button.isChecked() == True:
            self.court.canvas.on_start()
        elif self.sim_button.isChecked() == True:
            self.court.start_sim()


if __name__ == "__main__":
    import sys

    print("Starting")
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

