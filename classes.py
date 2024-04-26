import math

class selector:

    def __init__(self):
        """
        Initialize the object with default mode and aspect values.
        """
        self._mode = "selector"
        self._aspect = "selector"

    def change_mode(self, mode):
        """
        Set the mode and the aspect of the object.
        change_mode (str): The new mode to set.
        """
        self._mode = mode
        self._aspect = mode
        
    def create_object(self, x1, y1, x2, y2):
        """
        Create an object based on the specified mode and coordinates.
        
        Args:
            x1, y1, x2, y2: Coordinates of the points
        
        Returns:
            An object of the specified mode.
        
        Raises:
            ValueError: If the mode is not supported.
        """
        if self._mode == "line":

            return line(x1, y1, x2, y2)
        
        elif self._mode == "ruler":

            return ruler(x1, y1, x2, y2)
        
        elif self._mode == "selector":

            return selector()
        
        else:

            raise ValueError("Unsupported mode: {}".format(self._mode))
        

class point:

    def __init__(self, x, y):
        """
        Initialize the Point object with x and y coordinates.
        
        Parameters:
            x (int): x-coordinate of the point
            y (int): y-coordinate of the point
        """
        if x is None or y is None:

            raise ValueError("Point coordinates cannot be None")

        self._x = x
        self._y = y


class line:

    def __init__(self, point1, point2):
        """
        Initialize the Line object with two points.
        Parameters:
            point1 (tuple): Coordinates of the first point (x, y)
            point2 (tuple): Coordinates of the second point (x, y)
        """
        if point1 is None or point2 is None:

            raise ValueError("Point coordinates cannot be None")

        self._point1 = point1
        self._point2 = point2
        
class ruler(line):

    def __init__(self, point1, point2):
            """
            Initialize the Line object with two points and calculate the length of the line.

            Parameters:
                point1: tuple, the coordinates of the first point
                point2: tuple, the coordinates of the second point
            """
            super().__init__(point1, point2)
            self._length = math.sqrt((self._point1._x - self._point2._x)**2 + (self._point1._y - self._point2._y)**2)

    def length(self):
            """
            Return the length of the line.
            """
            return self._length

class area(line):
     
    def __init__(self, point1, point2):
        """
        Initialize the object with two points and set the points list.
         
        :param point1: The first point
        :param point2: The second point
        """
        super().__init__(point1, point2)
        self._points = [point1, point2]

    def add_point(self, point):
        """
        Add a point to the list of points.
        Parameters:
            - point: The point to be added to the list.
        """
        self._points.append(point)

    def get_area(self):
        """
        Calculate the area of the polygon defined by the points.
        
        Returns:
            float: The area of the polygon, or 0 if the points define a line or a point.
        """
        Area = 0
        number_of_points = len(self._points)
        if not number_of_points < 3:
   
            for i in range(0,number_of_points):
                if i+1 < number_of_points:
                    y2 = self._points[i+1]._y
                    y1 = self._points[i]._y
                    x2 = self._points[i+1]._x
                    x1 = self._points[i]._x
                    
                else:
                    y2 = self._points[0]._y
                    y1 = self._points[i]._y
                    x2 = self._points[0]._x
                    x1 = self._points[i]._x

                Area += (y1 - y2)*(x1 + x2)/2

        return abs(Area/2)

class angle:

    def __init__(self, point1, point2, point3):
        """
        Initialize the Angle object with three points.
        Parameters:
            point1 (tuple): Coordinates of the first point (x, y)
            point2 (tuple): Coordinates of the second point (x, y)
            vertex (tuple): Coordinates of the vertex (x, y)
        """
        if point1 is None or  point2 is None or point3 is None:
            raise ValueError("Point coordinates cannot be None")

        self._point1 = point1
        self._point2 = point2
        self._vertex = point3
    
    def get_angle(self):
        """
        Calculate the angle between two vectors formed by the points self._point1, self._point2, and self._vertex.
        Returns the angle in radians.
        """
        vector1 = [self._point1._x-self._vertex._x, self._point1._y-self._vertex._y]
        vector2 = [self._point2._x-self._vertex._x, self._point2._y-self._vertex._y]

        return math.acos( ( vector1._x*vector2._x + vector1._y*vector2._y )/( math.sqrt(vector1._x**2 + vector1._y**2)*math.sqrt(vector2._x**2 + vector2._y**2) ) )


class box_text:
     
    def __init__(self, point1, point2):
        """
        Initialize the object with two points, a default color, font, size, and an empty text.
        """
        self._point1 = point1
        self._point2 = point2
        self._color = "black"
        self._font = "Arial"
        self._size = 12
        self._text = ""

    def change_text(self, text):
        """
        Sets the text of the instance to the provided text parameter.

        Parameters:
            text (str): The new text to set.
        """
        self._text = text

    def change_color(self, color):
        """
        A function to change the color attribute to the specified color.
        
        Parameters:
            color (str): The new color to set for the object.

        """
        self._color = color

    def change_font(self, font):
        """
        Change the font used by the object.

        Parameters:
            font (str): The new font to set.
        """
        self._font = font

    def change_size(self, size):
        """
        Set the size of the object to the specified size.

        Parameters:
            size (int): The new size to set for the object.
        """
        self._size = size

r = math.sqrt(3)
lado = 1
p1 = point(0,0)
p2 = point(lado,0)
p3 = point(3/2*lado,r/2*lado)
p4 = point(lado,r*lado)
p5 = point(0,r*lado)
p6 = point(-1/2*lado,r/2*lado)

r1 = ruler(p1,p2)
r2 = ruler(p1,p3)
r3 = ruler(p1,p4)
r4 = ruler(p1,p5)
r5 = ruler(p1,p6)
print(r1.length(),r2.length(),r3.length(),r4.length(),r5.length())
a =  area(p1,p2)
a.add_point(p3)
a.add_point(p4)
a.add_point(p5)
a.add_point(p6)
print(a.get_area())