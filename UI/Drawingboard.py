import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt,QLineF
from PyQt5.QtGui import QPixmap,QPen
from Players import MovableDisk
from Ball import BallDisk
import SavePos

if hasattr(Qt,'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
if hasattr(Qt,'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,True)

class Scene_(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.clear()
        s_w = 570
        s_h = 580
        self.ball_placed  = False
        self.setSceneRect(0, 0, s_w, s_h)
        self.initScene(s_w,s_h)

    def initScene(self,s_w,s_h):
        self.clear()

        self.ball = BallDisk(0, 250, 15, self)
        self.disk = MovableDisk(0, 200, 30, self,self.ball)
        self.disk2 = MovableDisk(0, 150, 30, self,self.ball)
        self.disk3 = MovableDisk(0, 100, 30, self,self.ball)
        self.disk4 = MovableDisk(0, 50, 30, self,self.ball)
        self.disk5 = MovableDisk(0, 0, 30, self,self.ball)

        pic = QPixmap("./images/court.png")
        pic = pic.copy(470,0,
                       470,500)
        #pic = pic_copy.scaled(s_w, s_h, Qt.KeepAspectRatio, Qt.FastTransformation)

        img = self.addPixmap(pic)
        img.setPos(50,40)

        self.addItem(self.disk)
        self.addItem(self.disk2)
        self.addItem(self.disk3)
        self.addItem(self.disk4)
        self.addItem(self.disk5)
        self.addItem(self.ball)

        self.disk.addText(5)
        self.disk2.addText(4)
        self.disk3.addText(3)
        self.disk4.addText(2)
        self.disk5.addText(1)

    def clear_c(self):
        self.initScene(570,580)

    def setDraw(self):
        if self.ball_placed == False:
            print("Place ball!")
        else:
            self.disk.set_draw()
            self.disk2.set_draw()
            self.disk3.set_draw()
            self.disk4.set_draw()
            self.disk5.set_draw()
            self.ball.set_draw()

    def showPos(self):
        print("P1 ", self.disk.getPos())
        print("P2 ", self.disk2.getPos())
        print("P3 ", self.disk3.getPos())
        print("P4 ", self.disk4.getPos())
        print("P5 ", self.disk5.getPos())
        print("Ball", self.ball.getPos())
        print("Seg = ", self.ball.seg)

    def savePos(self):
        SavePos.save_pos(self.ball.segData)

    def SegPos(self):

        p1 = self.disk.tempTraj
        print("p1:", p1)
        if not p1:
            print("Getting")
            p1 = self.disk.get_pos()

        p2 = self.disk2.tempTraj
        if not p2:
            p2 = self.disk2.get_pos()

        p3 = self.disk3.tempTraj
        if not p3:
            p3 = self.disk3.get_pos()

        p4 = self.disk4.tempTraj
        if not p4:
            p4 = self.disk4.get_pos()

        p5 = self.disk5.tempTraj
        if not p5:
            p5 = self.disk5.get_pos()

        self.ball.addSeg(p1,p2,p3,p4,p5)

        self.disk.clear_traj()
        self.disk2.clear_traj()
        self.disk3.clear_traj()
        self.disk4.clear_traj()
        self.disk5.clear_traj()

    def mousePressEvent(self, e:QGraphicsSceneMouseEvent):
        self.x_s = e.scenePos().x()
        self.y_s = e.scenePos().y()
        if self.disk.if_selected() == True or self.disk2.if_selected() == True or self.disk3.if_selected() == True or self.disk4.if_selected() == True or  self.disk5.if_selected() == True:
            pass
        elif self.disk.allow_draw == False or self.disk2.allow_draw == False or self.disk3.allow_draw == False or self.disk4.allow_draw == False or self.disk5.allow_draw == False:
            pass
        elif self.disk.if_selected() == False and self.disk2.if_selected() == False and self.disk3.if_selected() == False and self.disk4.if_selected() == False and self.disk5.if_selected() == False:
            print("Ready")
            self.ball.tmpData.append([-1,-1])
            self.ball.tmpData.append(self.ball.getPos())

            self.ball.seg += 1
            #self.SegPos()

        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e:QGraphicsSceneMouseEvent):
        if self.disk.if_selected() == True or self.disk2.if_selected() == True or self.disk3.if_selected() == True or self.disk4.if_selected() == True or  self.disk5.if_selected() == True:
            pass
        elif self.disk.allow_draw == False or self.disk2.allow_draw == False or self.disk3.allow_draw == False or self.disk4.allow_draw == False or  self.disk5.allow_draw == False:
            pass
        else :
            x_e = e.scenePos().x()
            y_e = e.scenePos().y()
            pen = QPen(Qt.darkGreen, 3, Qt.DotLine)
            line = QGraphicsLineItem(QLineF(self.x_s,self.y_s,x_e,y_e))
            line.setPen(pen)
            self.addItem(line)

            self.ball.set_pos(x_e-12.5,y_e-12.5)

            p1 = self.disk.ori_pos()
            p2 = self.disk2.ori_pos()
            p3 = self.disk3.ori_pos()
            p4 = self.disk4.ori_pos()
            p5 = self.disk5.ori_pos()

            self.ball.place_nearest(p1, p2, p3, p4, p5)

            #self.ball.posdata.append(self.ball.getPos())
            self.ball.made_pass = True
            print("pass:",self.ball.getPos())

            if self.ball.Basket_shot == True:
                self.ball.tmpData.append(self.ball.shot_made())
            else:
                self.ball.tmpData.append(self.ball.getPos())

            #self.ball.pass_(self.ball.posdata)
            self.ball.posdata = []

            print("PASS!")
            self.SegPos()

        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e:QGraphicsSceneMouseEvent):
        x_e = e.scenePos().x()
        y_e = e.scenePos().y()
        #if self.disk.if_selected() == True or self.disk2.if_selected() == True or self.disk3.if_selected() == True or self.disk4.if_selected() == True or  self.disk5.if_selected() == True:
        #    print("Player selected")
        if self.disk.allow_draw == False or self.disk2.allow_draw == False or self.disk3.allow_draw == False or self.disk4.allow_draw == False or self.disk5.allow_draw == False:
            print("Place ball")
            self.ball_placed = True
            self.ball.set_pos(x_e-12.5,y_e-12.5)
            self.ball.getPos()

            p1 = self.disk.ori_pos()
            p2 = self.disk2.ori_pos()
            p3 = self.disk3.ori_pos()
            p4 = self.disk4.ori_pos()
            p5 = self.disk5.ori_pos()

            self.ball.place_nearest(p1,p2,p3,p4,p5)


        super().mouseDoubleClickEvent(e)

class App(QGraphicsView):
    def __init__(self,parent = None):
        super(App,self).__init__(parent)
        self.adjustSize()
        self.scene = Scene_()
        self.setScene(self.scene)


        self.setFocusPolicy(Qt.NoFocus)

        self.setSizePolicy(QSizePolicy.Fixed,
                           QSizePolicy.Fixed)

        self.setStyleSheet("border: transparent;")


