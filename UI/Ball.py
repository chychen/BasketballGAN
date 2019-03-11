from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt,QPointF
import numpy as np
from math import sqrt
import Bezier

class BallDisk(QGraphicsEllipseItem):
    def __init__(self,x,y,radius_,scene):
        super().__init__(0,0,radius_,radius_)
        self.scene = scene
        self.posdata = []
        self.tmpData = []
        self.setPos(x,y)
        self.setBrush(Qt.green)

        self.setFlag(QGraphicsItem.ItemIsMovable,False)
        self.setAcceptHoverEvents(False)
        self.setSelected(False)

        self.Basket_shot = False

        self.allow_draw = False
        self.segData = []
        self.seg = 0
        self.radius = radius_/2

        self.pen = QPen(Qt.black,3,Qt.SolidLine)

        self.c_click = True
        self.hide()

    def shot_made(self):
        x = float(455 + 470 + self.radius - 20 - 7.5)
        y = float(288 + self.radius - 40 - 7.5)
        pos = [x, y]
        self.Basket_shot = False

        return pos

    def getPos(self):
        x = float(self.pos().x())+470+self.radius-20-7.5
        y = float(self.pos().y())+self.radius-40-7.5
        pos = [x,y]

        return pos

    def getOriPos(self):
        x = float(self.pos().x())
        y = float(self.pos().y())
        pos = [x, y]

        return pos

    def pass_(self,pos):
        print(self.tmpData)
        #self.tmpData = pos
        #self.tmpData.append(pos)

    def addSeg(self,p1,p2,p3,p4,p5):
        segData = []
        print(self.tmpData)
        if self.tmpData[0] == [-1,-1]:
            self.tmpData = ([self.tmpData[1]]*2)+self.tmpData
            print(self.tmpData)
            print("No data")
        segData.append(self.tmpData)
        segData.append(p1)
        segData.append(p2)
        segData.append(p3)
        segData.append(p4)
        segData.append(p5)

        #seg = np.array(segData)
        self.segData.append(segData)
        seg = np.array(self.segData)
        self.tmpData = [self.getPos()]
        print("segData:")
        print(seg.shape)


    def savePos(self):
        self.pen_type = 1
        print("ball")
        print(self.pos)
        posdata = np.array(self.posdata)

        data = np.array(posdata[0])
        for i in range(1,len(posdata)):
            data = np.vstack((data,posdata[i]))
        print(data)
        print(data.shape)
        data[:,0] = data[:,0]+470

        nx,ny = Bezier.plotB(data, nTimes=100)
        x = np.array(nx)
        y = np.array(ny)

        point = np.column_stack([x, y])
        point = point.reshape((100, 2))
        point = np.divide(point,10)
        point = point[::-1]
        print(point)

        return point

    def set_pos(self,x,y):
        self.setPos(x,y)
        self.show()

    def set_draw(self):
        if self.allow_draw is False:
            self.allow_draw = True
            x = float(self.pos().x())
            y = float(self.pos().y())
            self.tmpData.append([x+470+self.radius-20 , y+self.radius-40])

        elif self.allow_draw is True:
            self. allow_draw = False

    def hoverEnterEvent(self,event:'QGraphicsSceneHoverEvent'):
        #when Cursor enters the object, set cursor to open hand
        event.ignore()
        pass

    def hoverLeaveEvent(self,event:'QGraphicsSceneHoverEvent' ):
        #when Cursor leaves the object, retore mouse cursor
        QApplication.instance().setOverrideCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self,event:'QGraphicsSceneMouseEvent'):
        event.ignore()
        pass

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        self.cursorStartposition = event.scenePos()
        if self.c_click:
            self.start = QPointF(self.cursorStartposition.x(),self.cursorStartposition.y())
            self.c_click = False

    def mouseDoubleClickEvent(self, event: 'QGraphicsSceneMouseEvent'):
        pass

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.allow_draw:
            print("hi")

    def calcDistance(self,p2):
        d = sqrt( (self.getOriPos()[0]-p2[0])**2 + (self.getOriPos()[1]-p2[1])**2 )

        return d

    def place_nearest(self,p1,p2,p3,p4,p5):
        pos = []
        pos.append(p1)
        pos.append(p2)
        pos.append(p3)
        pos.append(p4)
        pos.append(p5)
        basket = [458,281]
        pos.append(basket)

        player1 = self.calcDistance(p1)
        player2 = self.calcDistance(p2)
        player3 = self.calcDistance(p3)
        player4 = self.calcDistance(p4)
        player5 = self.calcDistance(p5)
        basketd = self.calcDistance(basket)

        tmp = []
        tmp.append(player1)
        tmp.append(player2)
        tmp.append(player3)
        tmp.append(player4)
        tmp.append(player5)
        tmp.append(basketd)

        p = tmp.index(min(tmp))

        if p == 5:
            self.Basket_shot = True

        self.set_pos(pos[p][0],pos[p][1])








