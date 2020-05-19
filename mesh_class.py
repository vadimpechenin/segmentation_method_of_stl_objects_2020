import numpy as np

class Mesh_class():
    """Класс slt-объекта"""
    def __init__(self,num_vertices,num_faces):
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.vertices =0
        self.faces = 0
        self.normals =0
        self.Cmin = 0
        self.Cmax = 0