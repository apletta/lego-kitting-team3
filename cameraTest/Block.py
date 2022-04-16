import numpy
class Block:
    """"
    x: CENTROID OF X POSITION IN BASE_LINK {m}
    y: CENTROID OF Y POSITION IN BASE_LINK {m}
    length: defined as dimension which is greater than width {m}
    width: defined as dimensions which is shorter than length {m}
    angle: angle of length of block measured from the x "axis" in the camera frame {rad}
    color: STRING OF "BLUE" OR "RED"
    """
    def __init__(self,x_,y_,length_,width_,angle_,color_) -> None:
        self.x = x_ # Center x position, float
        self.y = y_ # CXenter y position, float
        if(length_ > width_):
            self.length = length_ # length, float; should be greater than or equal to width
            self.width = width_ # width, float; should be smaller than or equal to length
        else:
            self.length = width_
            self.width = length_
        self.angle = angle_ # angle, in radians
        self.color = color_ # color, string
