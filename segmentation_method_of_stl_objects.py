""" Метод сегментации stl, основанный на тензорах кривизны по
 статьям
 1. Lavou, Guillaume. A new curvature tensor based segmentation method for optimized
 triangulated CAD meshes - основной алгоритм сегментации
 2. Mark Meyer. Discrete Differential-Geometry Operators
 for Triangulated 2-Manifolds- вычисление кривизн 1 и 2 в вершинах
 Версия для MATLAB Segmentation_by_curvature_tensor_Universal_2020.m
"""
# Импорт библиотек
import numpy as np

#Использование библеотеки trimesh 3.5.22
import trimesh

#Импорт основных переменных метода
from main_variables_for_segmentation import Main_variables
#Импорт вспомогательных функций
import supporting_functions_of_segmentation as sff
# Класс импорта stl
from import_stl_data import Import_stl_data

ex_m_var=Main_variables()

stl_data=Import_stl_data(ex_m_var.pl_zagr,ex_m_var.pl,ex_m_var.pl_sphere_cyl,ex_m_var.path_file)

mesh = stl_data.import_data()


