import numpy as np

class Main_variables():
    """Класс, в котором храняться главные переменные проекта"""
    def __init__(self):
        self.pl_zagr = np.array([0, 0, 0, 0, 1])
        """ Переменная выполнения этапов 1-5
            1 этап - загрузка структуры stl
            2 этап - вычисление кривизн поверхности
            3 этап - кластеризация вершин по кривизне, предварительная классификация фасет
            4 этап - предварительная сегментация фасет """
        self.pl = np.array([0, 0, 0, 0, 1, 1])  # Прорисовка этапов 1-5
        self.pl_sphere_cyl = np.array([6, 1])  # 1 - какая серия эксперментов; 2 - вид из серии экспериментов
        # 1, 2 - серия с мед.дет., теория и практика; 1 - 1 Деталь; 2 - 5 Деталь Олеса Рус; 3 - 4 Деталь; 4 - 2 деталь;
        # 3, 4 - серия с лопатками, теория и практика; 1 - единичная лопатка РК турбины; 2 - блок из 3 лопаток СА
        # Blade_turbine_ideal, Лопатка_турбины_август_1_обр,40.412.007_идеал, 40.412.007_упрощенная_1
        # 5 - 3 сферы, теоретический пример. Three_spheres_radius64148
        self.pl_klast = 1  # какой алгоритм кластеризации использовать, k-mean или DBSCAN
        self.curvature_tolerance = 0.6  # допуск на кривизну при кластеризации
        self.angleTolerance = 30
        self.path_file = 'D:\\PYTHON\\Programms\\Segmentation_ex_2\\Segmentation_stl_py_2020\\results\\'