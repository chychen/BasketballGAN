from PyQt5.QtGui import QPen, QPolygonF
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt,QPointF,QLineF
from rdp import rdp
import numpy as np
from math import radians, atan2, sqrt,sin,cos
import Bezier
#court size x = 47 y = 50

class MovableDisk(QGraphicsEllipseItem):
    def __init__(self,x,y,radius_,scene,ball):
        super().__init__(0,0,radius_,radius_)
        self.scene = scene
        self.tempTraj = []
        self.wavetraj = []

        self.ball = ball
        self.radius = radius_/2
        self.last_traj = []
        self.last_rdp = []
        self.made_pass = False

        self.setPos(x,y)
        self.setBrush(Qt.red)
        self.setFlag(QGraphicsItem.ItemIsMovable,False)
        self.setAcceptHoverEvents(True)
        self.allow_draw = False
        self.selected = False
        self.has_ball = False

        self.pen_type = 1
        self.pen = QPen(Qt.black,3,Qt.SolidLine)

        self.c_click = True
        self.lines = QGraphicsItemGroup()
        self.lines.setZValue(-1)
        self.group = QGraphicsItemGroup()
        self.group.setZValue(-1)
        self.rdpGroup = QGraphicsItemGroup()
        self.temp_group = QGraphicsItemGroup()

        self.io = QGraphicsTextItem()
        self.io.setAcceptHoverEvents(False)
        self.io.setTabChangesFocus(False)

    def addText(self,num):
        self.io.setPlainText("{}".format(num))
        self.io.setDefaultTextColor(Qt.white)
        self.io.setPos(self.pos().x()+6,self.pos().y()+2)
        self.scene.addItem(self.io)

    def if_selected(self):
        return self.selected

    def get_pos(self):
        x = (self.pos().x())+self.radius+470-20
        y = (self.pos().y())+self.radius-40

        point =[[x,y]]*2
        #point = point.reshape((1, 2))
        #print(point)

        return point

    def ori_pos(self):
        x = (self.pos().x()) + self.radius
        y = (self.pos().y()) + self.radius

        point = [x, y]
        # point = point.reshape((1, 2))
        # print(point)

        return point

    def clear_traj(self):
        self.tempTraj = []
        self.wavetraj = []

    def setInitPos(self,x,y):
        self.tempTraj.append([x+470+self.radius-20 , y+self.radius-40])

    def savePos(self):
        nx,ny = Bezier.plotB(self.rdata, nTimes=100)
        x = np.array(nx)
        y = np.array(ny)

        point = np.column_stack([x, y])
        point = point.reshape((100, 2))
        point = np.divide(point,10)
        point = point[::-1]
        print(point)

        return point

    def set_draw(self):
        if self.allow_draw is False:
            self.allow_draw = True
            self.scene.addItem(self.rdpGroup)
            self.scene.addItem(self.group)
            self.scene.addItem(self.temp_group)
            print("Pos:",self.pos())
            x = float(self.pos().x())
            y = float(self.pos().y())
            self.setInitPos(x,y)

        elif self.allow_draw is True:
            self. allow_draw = False

    def hoverEnterEvent(self,event:'QGraphicsSceneHoverEvent'):
        #Cursor enters the object, set cursor to open hand
        QApplication.instance().setOverrideCursor(Qt.OpenHandCursor)
        self.selected = True

    def hoverLeaveEvent(self,event:'QGraphicsSceneHoverEvent' ):
        #when Cursor leaves the object, retore mouse cursor
        QApplication.instance().setOverrideCursor(Qt.ArrowCursor)
        self.selected = False

    def mouseMoveEvent(self,event:'QGraphicsSceneMouseEvent'):
        #move object with mouse
        new_cursor_position = event.scenePos()
        old_cursor_position = event.lastPos()
        new_x = new_cursor_position.x()-old_cursor_position.x()
        new_y = new_cursor_position.y()-old_cursor_position.y()
        self.io.setPos(self.pos().x() + 6, self.pos().y() + 2)

        self.setPos(QPointF(new_x, new_y))

        if self.allow_draw:
            self.wavetraj.append( [ new_x+self.radius , new_y+self.radius])
            self.tempTraj.append([new_x+self.radius+470-20,new_y+self.radius-40])
            self.last_traj.append([new_x+self.radius,new_y+self.radius])

            self.lines = QGraphicsItemGroup()
            '''
            p1 = QGraphicsEllipseItem(new_x+self.radius,new_y+self.radius, 10, 10)
            p1.setBrush(Qt.red)
            self.scene.addItem(p1)
            '''
            l_rdata = rdp(self.last_traj, epsilon=1.5)
            l_rdata = np.array(l_rdata)

            #self.ball_pos = l_rdata
            self.last_rdp = np.array(l_rdata)
            l = len(l_rdata) - 2

            for i in range(l):
                line = QGraphicsLineItem(QLineF(self.last_rdp[i, 0], self.last_rdp[i, 1], self.last_rdp[i + 1, 0], self.last_rdp[i + 1, 1]))
                line.setPen(self.pen)
                self.lines.addToGroup(line)

            self.group.addToGroup(self.lines)
            self.group.removeFromGroup(self.lines)

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        QApplication.instance().setOverrideCursor(Qt.ClosedHandCursor)
        #self.tempTraj = []
        self.cursorStartposition = event.scenePos()
        if self.c_click:
            self.start = QPointF(self.cursorStartposition.x(),self.cursorStartposition.y())
            self.c_click = False

    def mouseDoubleClickEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.allow_draw:
            if self.pen_type == 1:
                self.temp_group.removeFromGroup(self.lines)

                wavex, wavey = Bezier.plotB(self.last_rdp[:-1], 20)

                x = np.array(wavex)
                y = np.array(wavey)

                point = np.column_stack([x, y])
                point = point.reshape((20, 2))

                self.lines = QGraphicsItemGroup()

                for i in range(len(point) - 1):
                    wave = self.calcWiggle(point[i], point[i + 1], wavelength=10, waveheight=10)

                    l = len(wave) - 2
                    for i in range(l):
                        line = QGraphicsLineItem(QLineF(wave[i][0], wave[i][1], wave[i + 1][0], wave[i + 1][1]))
                        line.setPen(self.pen)
                        self.lines.addToGroup(line)

                    self.temp_group.addToGroup(self.lines)

                self.wavetraj = []
                l = len(self.last_rdp)-1

                self.temp_group.addToGroup(self.lines)
                self.ball.set_pos(self.last_rdp[l,0],self.last_rdp[l,1])
                self.ball.tmpData.extend(self.tempTraj)

                self.ball.seg += 1
                self.has_ball = True

                self.pen_type = 1
                self.pen = QPen(Qt.black, 3, Qt.SolidLine)

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        QApplication.instance().setOverrideCursor(Qt.OpenHandCursor)
        if self.allow_draw:
            self.last_traj = []
            self.temp_group.addToGroup(self.lines)

            '''
            startP = self.tempTraj[-1]
            endP = self.tempTraj[-2]
            polygon = QPolygonF()

            rightP,leftP = ArrowHead.arrowHead(startP,endP,20,0.5)
            polygon.append(QPointF(rightP[0],rightP[1]))
            polygon.append(QPointF(leftP[0],leftP[1]))
            polygon.append(QPointF(endP[0],endP[1]))

            lines = QGraphicsItemGroup()
            line = QGraphicsLineItem(QLineF(rightP[0],rightP[1] , endP[0], endP[1]))
            line.setPen(self.pen)
            lines.addToGroup(line)

            line = QGraphicsLineItem(QLineF(leftP[0], leftP[1], endP[0], endP[1]))
            line.setPen(self.pen)
            poly = QGraphicsPolygonItem(polygon)
            poly.setBrush(Qt.red)
            lines.addToGroup(line)
            self.scene.addItem(poly)

        self.temp_group.addToGroup(self.lines)
        '''

    def calcAngle(self,p1,p2):
        return atan2( (p2[1]-p1[1]) , (p2[0]-p1[0]) )

    def calcDistance(self,p1,p2):
        return sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

    def calcWiggle(self,p1,p2,wavelength,waveheight,curveSqr = 0.57):
        diagonal = self.calcDistance(p1,p2)
        angleRad = self.calcAngle(p1,p2)

        waves = diagonal//int(wavelength)

        if waves == 0:
            waves = 1
        waveInterval = diagonal/float(waves)

        maxBcpLength = sqrt((waveInterval/4.0)**2 + (waveheight/2.0)**2)
        bcpLength = maxBcpLength*curveSqr
        bcpInclination = self.calcAngle([0,0],[waveInterval/4., waveheight/2. ])

        wigglePoints = [p1]
        prevFlexPt = p1
        polarity = 1

        for waveIndex in range(0,int(waves*2)):
            bcpOutAngle = (angleRad + bcpInclination) *polarity

            bcpOut = [ prevFlexPt[0]+cos(bcpOutAngle)*bcpLength , prevFlexPt[1]+sin(bcpOutAngle)*bcpLength]

            flexPT = [ (prevFlexPt[0]+cos(angleRad)*waveInterval/2.) , (prevFlexPt[1]+sin(angleRad)*waveInterval/2.) ]

            bcpInAngle = angleRad+(radians(180)-bcpInclination)*polarity

            bcpIn = [ flexPT[0]+cos(bcpInAngle)*bcpLength , flexPT[1]+sin(bcpInAngle)*bcpLength ]

            wigglePoints.append(bcpOut)
            wigglePoints.append(bcpIn)
            wigglePoints.append(flexPT)

            polarity *= -1
            prevFlexPt = flexPT

        return wigglePoints




