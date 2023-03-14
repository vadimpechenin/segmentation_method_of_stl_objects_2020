import numpy as np

class Main_variables():
    """Класс, в котором храняться главные переменные проекта"""
    def __init__(self):
        self.pl_zagr = np.array([0, 0, 0, 0, 0])
        """ Переменная выполнения этапов 1-5
            1 этап - загрузка структуры stl
            2 этап - вычисление кривизн поверхности
            3 этап - кластеризация вершин по кривизне, предварительная классификация фасет
            4 этап - предварительная сегментация фасет 
            5 этап - окончательная сегментация фасет
            6 этап - визуализация без мелких деталей"""
        self.pl = np.array([0, 0, 0, 1, 1, 1])  # Прорисовка этапов 1-5
        self.pl_sphere_cyl = np.array([5, 1])  # 1 - какая серия эксперментов; 2 - вид из серии экспериментов
        self.pl_klast = 1  # какой алгоритм кластеризации использовать, k-mean или DBSCAN
        self.curvature_tolerance = 0.6  # допуск на кривизну при кластеризации
        self.angleTolerance = 30
        self.path_file = 'D:\\PYTHON\\Programms\\Segmentation_ex_2\\Segmentation_stl_py_2020\\results\\'
        #self.path_file = 'D:\\Vadim\\PYTHON\Programms\\Segmentation_ex_2\\Segmentation_stl_py_2020\\results\\'