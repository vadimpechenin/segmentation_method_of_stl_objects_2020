"""Программа основана на вычислении кривизн по коду D.Kroon University of Twente (August 2011)"""
import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy

def calculation_curvature_tensor(mesh, pl_zagr, pl, pl_sphere_cyl, path_file):
    """Функция для расчета главных кризивн stl объекта"""
    name_safe = sff.name_of_results(pl_sphere_cyl)+'_stage_1'
    if (pl_zagr[1] == 1):
        pass
    else:
        curv_load_mat = scipy.io.loadmat(path_file + name_safe + '.mat')
        mesh.Cmin = np.array(curv_load_mat['Cmin'])
        mesh.Cmax = np.array(curv_load_mat['Cmax'])

    if pl[1] == 1:
        # mesh.show()
        struct_seg = np.array([1])
        num_segments = np.array([1])
        color_segmetns = np.array([1])
        # Многомерную матрицу зададим в виде двумерного списка, у которого в ячейках будут двумерные матрицы
        surface_seg = []
        nr = 1  # количество строк
        nc = 1  # количество столбцов
        for r in range(nr):
            surface_seg.append([])
            for c in range(nc):
                surface_seg[r].append([])
                surface_seg[r][c].append(mesh.faces)  # добавляем очередной элемент в строку
        title = 'Загруженный для сегментации объект stl '
        sff.plot_stl_color(struct_seg, num_segments, color_segmetns, surface_seg, mesh.vertices, title)
    return mesh