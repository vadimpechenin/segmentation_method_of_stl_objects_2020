"""Программа основана на вычислении кривизн по коду D.Kroon University of Twente (August 2011)"""
import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy

def calculation_curvature_tensor(mesh, pl_zagr, pl, pl_sphere_cyl, path_file):
    """Функция для расчета главных кризивн stl объекта"""
    name_safe = sff.name_of_results(pl_sphere_cyl)+'_stage_1'
    if (pl_zagr[1] == 1):
        if (1 == 0):
            # Временная замена для отладки функции
            mesh_load_mat = scipy.io.loadmat(path_file + 'sphere_fv_5.mat')
            v = np.array(mesh_load_mat['v'])
            f = np.array(mesh_load_mat['f'])
            for j in range(f.shape[0]):
                for i in range(f.shape[1]):
                    f[j, i] = f[j, i] - 1
            mesh1 = Mesh_class(v.shape[0], f.shape[0])
            mesh1.vertices = copy.deepcopy(v)
            mesh1.faces = copy.deepcopy(f)
            Umin, Umax, Cmin, Cmax, Cmean, Cgaussian = sff.patchcurvature_2014(mesh1)
        else:
            Umin, Umax, Cmin, Cmax, Cmean, Cgaussian = sff.patchcurvature_2014(mesh)
        # Сохранение данных по кривизне
        scipy.io.savemat(path_file + name_safe + '.mat', {'Umin': Umin,
                                                          'Umax': Umax,
                                                          'Cmin': Cmin,
                                                          'Cmax': Cmax,
                                                          'Cmean': Cmean,
                                                          'Cgaussian': Cgaussian})
    else:
        mesh_curve_load_mat = scipy.io.loadmat(path_file + name_safe + '.mat')
        Cmin, Cmax = np.array(mesh_curve_load_mat['Cmin']), np.array(mesh_curve_load_mat['Cmax'])
    mesh.Cmin = copy.deepcopy(Cmin)
    mesh.Cmax = copy.deepcopy(Cmax)

    # Прорисовка решения (карта кривизны вершин)
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