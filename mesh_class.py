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
        #Структура сегментированныой сетки
        # В виде списков с вложенными матрицами
        # Структура фасет сегментов
        self.surface_seg = 0
        # Структура нормалей фасет сегментов
        self.surfaceNormal_seg = 0
        # Структура кривизн фасет сегментов
        self.surfaceCurve_seg = 0
        # В виде матриц
        # Площади сегментов
        self.area_segments = 0
        # Количество фасет в каждом сегменте
        self.num_segments = 0
        # Общая кривизна каждого сегмента
        self.curve_of_segments = 0
        # Число, отвечающее за цвет каждого сегмента
        self.color_segments = 0
        # Количество сегментов
        self.struct_seg = 0
