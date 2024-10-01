import pygame  
from pygame.locals import *  
from OpenGL.GL import *  
from OpenGL.GLU import *  
import math  
import numpy as np  
  
# Constants  
WIDTH, HEIGHT = 800, 600  
FOV = 60  
NEAR_PLANE = 0.1  
FAR_PLANE = 100.0  
MOVE_SPEED = 0.1  
ROTATE_SPEED = 0.05  
  
# Camera class to handle first-person view  
class Camera:  
    def __init__(self):  
        self.pos = np.array([0.0, 0.0, 0.0])  
        self.direction = np.array([0.0, 0.0, -1.0])  # Looking forward  
        self.up = np.array([0.0, 1.0, 0.0])  
        self.right = np.cross(self.direction, self.up)  
  
    def move_forward(self, distance):  
        self.pos += self.direction * distance  
  
    def move_backward(self, distance):  
        self.pos -= self.direction * distance  
  
    def move_right(self, distance):  
        self.pos += self.right * distance  
  
    def move_left(self, distance):  
        self.pos -= self.right * distance  
  
    def rotate_right(self, angle):  
        cos_angle = math.cos(angle)  
        sin_angle = math.sin(angle)  
        self.direction = np.dot(self.direction, [cos_angle, 0, sin_angle])  
        self.direction = np.normalize(self.direction)  
        self.right = np.cross(self.direction, self.up)  
  
    def rotate_left(self, angle):  
        self.rotate_right(-angle)  
  
    def update_view(self):  
        glLoadIdentity()  
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],  
                  self.pos[0] + self.direction[0], self.pos[1] + self.direction[1], self.pos[2] + self.direction[2],  
                  self.up[0], self.up[1], self.up[2])  
  
# Initialize Pygame and OpenGL  
pygame.init()  
display = (WIDTH, HEIGHT)  
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)  
glEnable(GL_DEPTH_TEST)  
gluPerspective(FOV, (display[0] / display[1]), NEAR_PLANE, FAR_PLANE)  
  
# Initialize camera  
camera = Camera()  
  
# Main loop  
running = True  
clock = pygame.time.Clock()  
while running:  
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  
      
    # Update camera view  
    camera.update_view()  
  
    # Draw house (simple cube representing the boundaries, centered at origin)  
    # Add some color to the cube for visibility  
    glColor3f(1.0, 0.0, 0.0)  # Set color to red  
    glBegin(GL_QUADS)  
    # Front face  
    glVertex3f(-1.0, -1.0, 1.0)  
    glVertex3f(1.0, -1.0, 1.0)  
    glVertex3f(1.0, 1.0, 1.0)  
    glVertex3f(-1.0, 1.0, 1.0)  
    # Back face  
    glVertex3f(-1.0, -1.0, -1.0)  
    glVertex3f(-1.0, 1.0, -1.0)  
    glVertex3f(1.0, 1.0, -1.0)  
    glVertex3f(1.0, -1.0, -1.0)  
    # Left face  
    glVertex3f(-1.0, -1.0, -1.0)  
    glVertex3f(-1.0, -1.0, 1.0)  
    glVertex3f(-1.0, 1.0, 1.0)  
    glVertex3f(-1.0, 1.0, -1.0)  
    # Right face  
    glVertex3f(1.0, -1.0, -1.0)  
    glVertex3f(1.0, 1.0, -1.0)  
    glVertex3f(1.0, 1.0, 1.0)  
    glVertex3f(1.0, -1.0, 1.0)  
    # Bottom face  
    glVertex3f(-1.0, -1.0, -1.0)  
    glVertex3f(1.0, -1.0, -1.0)  
    glVertex3f(1.0, -1.0, 1.0)  
    glVertex3f(-1.0, -1.0, 1.0)  
    # Top face  
    glVertex3f(-1.0, 1.0, -1.0)  
    glVertex3f(-1.0, 1.0, 1.0)  
    glVertex3f(1.0, 1.0, 1.0)  
    glVertex3f(1.0, 1.0, -1.0)  
    glEnd()  
  
    # Draw some debugging information (optional)  
    glColor3f(1.0, 1.0, 1.0)  # Set color to white  
    glRasterPos2f(-0.9, 0.9)  # Position for text  
    glutBitmapString(GLUT_BITMAP_HELVETICA_18, b"Camera Pos: (%.2f, %.2f, %.2f)" % tuple(camera.pos))  
      
    pygame.display.flip()  
  
    # Handle events  
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
            running = False  
        # ...（省略事件处理代码，与之前的代码相同）  
  
    clock.tick(60)  
  
pygame.quit()