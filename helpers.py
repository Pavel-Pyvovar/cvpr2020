import math
import os
import random

import cv2
import numpy as np
import pygame
from OpenGL.GL import *

DEFAULT_COLOR = (0, 255, 0)
OBJECT_SCALE = 100


def render(img, obj, projection, model, color: bool, col: int):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * OBJECT_SCALE
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if not color:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            mc = hex_to_rgb(col)
            mc = mc[::-1]
            cv2.fillConvexPoly(img, imgpts, mc)

    return img


def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def hex_to_rgb(col):
    if col == 1:
        def r1(): return random.randint(70, 120)
        def r2(): return random.randint(120, 180)
        def r3(): return random.randint(20, 90)
    else:
        def r1(): return random.randint(0, 10)
        def r2(): return random.randint(80, 120)
        def r3(): return random.randint(240, 255)
    hex_color = '#%02X%02X%02X' % (r1(), r2(), r3())
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


class OBJ:
    generate_on_init = True

    @classmethod
    def loadTexture(cls, imagefile):
        surf = pygame.image.load(imagefile)
        image = pygame.image.tostring(surf, 'RGBA', 1)
        ix, iy = surf.get_rect().size
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        return texid

    @classmethod
    def loadMaterial(cls, filename):
        contents = {}
        mtl = None
        dirname = os.path.dirname(filename)

        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                # load the texture referred to by this declaration
                mtl[values[0]] = values[1]
                imagefile = os.path.join(dirname, mtl['map_Kd'])
                mtl['texture_Kd'] = cls.loadTexture(imagefile)
            else:
                mtl[values[0]] = list(map(float, values[1:]))
        return contents

    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.gl_list = 0
        dirname = os.path.dirname(filename)

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = self.loadMaterial(os.path.join(dirname, values[1]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
        if self.generate_on_init:
            self.generate()

    def generate(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()

    def render(self):
        glCallList(self.gl_list)

    def free(self):
        glDeleteLists([self.gl_list])
