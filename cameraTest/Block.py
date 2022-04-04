import numpy


class Block:
    """"
    x: CENTROID OF X POSITION IN BASE_LINK
    y: CENTROID OF Y POSITION IN BASE_LINK
    length: TO BE DESTROYED
    width: TO BE DESTORYED
    angle: TO BE DETERMINED 
    color: STRING OF "BLUE" OR "RED"
    
    """

    def __init__(self,x_,y_,length_,width_,angle_,color_) -> None:
        self.x = x_ # CENTROID X POSITION 
        self.y = y_ # CENTROID Y POSITION
        self.length = length_ #length of block
        self.width = width_ #width of block
        self.angle = angle_ # to be defined 
        self.color = color_ # STR "blue" or "red"