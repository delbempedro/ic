import math

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
        
    def create_object(self, point1, point2):
        """
        Create an object based on the current mode.

        Parameters:
            point1 (tuple): The coordinates of the first point.
            point2 (tuple): The coordinates of the second point.

        Returns:
            selector: If the mode is "selector".
            line: If the mode is "line".
            ruler: If the mode is "ruler".
            area: If the mode is "area".
            angle: If the mode is "angle".
            box_text: If the mode is "box_text".

        Raises:
            ValueError: If the mode is not supported.
        """
        if self._mode == "selector":

            return selector()
        
        elif self._mode == "line":

            return line(point1, point2)
        
        elif self._mode == "ruler":

            return ruler(point1, point2)
        
        elif self._mode == "area":

            return area(point1, point2)
        
        elif self._mode == "angle":

            return angle(point1, point2)
        
        elif self._mode == "box_text":

            return box_text(point1, point2)
        
        else:

            raise ValueError("Unsupported mode: {}".format(self._mode))


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
        self._area = 0 

    def add_point(self, point):
        """
        Add a point to the list of points.
        
        Parameters:
            - point: The point to be added to the list.
        
        This method appends the given point to the list of points.
        It then calculates the area of the polygon defined by the points.
        The area is updated and stored in the `_area` attribute of the object.
        
        Note: This method assumes that the points are in a plane and that they form a closed polygon.
        """
        self._points.append(point)
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

                self._area += (y1 - y2)*(x1 + x2)/2
        self._area = abs(self._area)

    def area(self):
        """
        Calculate the area of the polygon defined by the points.
        
        Returns:
            float: The area of the polygon, or 0 if the points define a line or a point.
        """
        return self._area

class angle:

    def __init__(self, point1, vertex):
        """
        Initialize the Angle object with three points.
        
        Parameters:
            point1 (tuple): Coordinates of the first point (x, y)
            point2 (tuple): Coordinates of the second point (x, y)
            vertex (tuple): Coordinates of the vertex (x, y)

        This method calculate the angle between two vectors formed by the points self._point1, self._point2, and self._vertex. 
        
        Returns:
            None.
        """
        if point1 is None or  vertex is None:
            raise ValueError("Point coordinates cannot be None")

        self._point1 = point1
        self._vertex = vertex

        #a,b = (float(i) for i in input().split(","))
        #self._point2 = point(a,b)#temporary solution to get the second point, in the future it should be a parameter defined by mouse click
        self._point2 = (0,0)
    
        vector1 = [self._point1._x-self._vertex._x, self._point1._y-self._vertex._y]
        vector2 = [self._point2._x-self._vertex._x, self._point2._y-self._vertex._y]

        self._radangle = math.acos( ( vector1[0]*vector2[0] + vector1[1]*vector2[1] )/( math.sqrt(vector1[0]**2 + vector1[1]**2)*math.sqrt(vector2[0]**2 + vector2[1]**2) ) )
        self._angle = math.degrees(self._radangle)

    def angle(self):
        """
        Return the angle of the object.

        Returns:
            float: The angle of the object.
        """
        return self._angle
    
    def radangle(self):
        """
        Return the radian angle of the object.

        Returns:
            float: The radian angle of the object.
        """
        return self._radangle

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
