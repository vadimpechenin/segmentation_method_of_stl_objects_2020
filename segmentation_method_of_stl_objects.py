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
# Класс вычисления главных кривизн
import сalculation_of_the_curvature_tensor as cct
import klastering_mesh_by_curvature_tensor as kct
ex_m_var=Main_variables()

stl_data=Import_stl_data(ex_m_var.pl_zagr,ex_m_var.pl,ex_m_var.pl_sphere_cyl,ex_m_var.path_file)
# 1 Этап. Загрузка структуры stl
mesh = stl_data.import_data()
# 2 Этап. Вычисления тензора кривизн (главных кривизн)
mesh=cct.calculation_curvature_tensor(mesh,ex_m_var.pl_zagr,ex_m_var.pl,ex_m_var.pl_sphere_cyl,ex_m_var.path_file)
# 3 Этап. Кластеризация данных по величинам главных кривизн. Сохранение структуры данных
idx_K, centers, massiv_face_klast,\
curvature_face_klast, curvature_mean =kct.klastering_vetices_of_mesh_by_curvature_tensor(mesh,ex_m_var.pl_zagr,
                                                                                         ex_m_var.pl,
                                                                                         ex_m_var.pl_sphere_cyl,
                                                                                         ex_m_var.path_file,
                                                                                         ex_m_var.curvature_tolerance)



