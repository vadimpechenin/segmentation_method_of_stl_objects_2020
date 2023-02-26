"""Программа для рисования результатов сегментации"""
# Импорт библиотек
import numpy as np
import scipy.io
import copy
from mesh_class import Mesh_class


path_file = 'D:\\Vadim\\PYTHON\Programms\\Segmentation_ex_2\\Segmentation_stl_py_2020\\results\\'
pl_sphere_cyl = np.array([1, 1])  # 1 - какая серия эксперментов; 2 - вид из серии экспериментов
        # 1, 2 - серия с мед.дет., теория и практика; 1 - 1 Деталь; 2 - 5 Деталь Олеса Рус; 3 - 4 Деталь; 4 - 2 деталь;
        # 3, 4 - серия с лопатками, теория и практика; 1 - единичная лопатка РК турбины; 2 - блок из 3 лопаток СА
        # Blade_turbine_ideal, Лопатка_турбины_август_1_обр,40.412.007_идеал, 40.412.007_упрощенная_1
        # 5 - 3 сферы, теоретический пример. Three_spheres_radius64148

#Импорт вспомогательных функций
import supporting_functions_to_plot as sfp

name_safe, struct_seg1 = sfp.name_of_results(pl_sphere_cyl)

mesh_load_mat=scipy.io.loadmat(path_file + name_safe+ '_matlab.mat')
vertices = np.array(mesh_load_mat['v'])
faces = np.array(mesh_load_mat['f'])
for j in range(faces.shape[0]):
    for i in range(faces.shape[1]):
        faces[j, i] = faces[j, i] - 1
normals = np.array(mesh_load_mat['n'])
num_vertices, num_faces = vertices.shape[0], faces.shape[0]
mesh = Mesh_class(num_vertices, num_faces)
mesh.vertices = copy.deepcopy(vertices)
mesh.faces = copy.deepcopy(faces)
mesh.normals = copy.deepcopy(normals)
struct_seg = np.array([1])
num_segments = np.array([1])
color_segmetns = np.array([1])
# Многомерную матрицу зададим в виде двумерного списка, у которого в ячейках будут двумерные матрицы
surface_seg1 = []
nr = 1  # количество строк
nc = 1  # количество столбцов
for r in range(nr):
    surface_seg1.append([])
    for c in range(nc):
        surface_seg1[r].append([])
        surface_seg1[r][c].append(mesh.faces)  # добавляем очередной элемент в строку
title = 'Загруженный для сегментации объект stl '
t=path_file + name_safe + '_paint.mat'
#sfp.plot_stl_color(struct_seg, num_segments, color_segmetns, surface_seg1, mesh.vertices, title)


segment_mat = scipy.io.loadmat(path_file + name_safe + '_paint.mat')
surface_seg1=segment_mat['surface_seg']
surfaceNormal_seg1=segment_mat['surfaceNormal_seg']
surface_seg = list(range(0, struct_seg1+1))
surfaceNormal_seg = list(range(0, struct_seg1+1))
for i in range(struct_seg1+1):
    if i==0:
        surface_seg[i] = mesh.faces
        surfaceNormal_seg[i] = mesh.normals
    else:
        surface_seg[i]= surface_seg1[0, i-1].astype('int64')
        surfaceNormal_seg[i] = surfaceNormal_seg1[0, i-1].astype('float64')
        for j in range(surface_seg[i].shape[0]):
            for jj in range(surface_seg[i].shape[1]):
                surface_seg[i][j, jj] = surface_seg[i][j, jj] - 1

sfp.plot_stl_faces_segmentation_paint(struct_seg1+1, num_segments, surface_seg,
                                        mesh.vertices, title)
g=0