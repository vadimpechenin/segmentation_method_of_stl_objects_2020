"""Модуль для кластеризации вершин по значениям их кривизн"""

import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy
import trimesh
from vtkplotter import trimesh2vtk, show


import matplotlib.pyplot as plt

def klastering_vetices_of_mesh_by_curvature_tensor(mesh,pl_zagr,pl,pl_sphere_cyl,path_file):
    # Кластеризация из книги Дж. Вандер Плас "Python для сложных задач. Наука о данных и машинное обучение"
    name_safe = sff.name_of_results(pl_sphere_cyl) + '_stage_2'
    # Количество кластеров
    N_clusters=sff.num_of_klasters(pl_sphere_cyl)
    if (pl_zagr[2] == 1):
        X = np.zeros([mesh.Cmin.shape[0], 2])

        # Нормирование для кластеризации
        Cmin1, Cmax1 = copy.deepcopy(mesh.Cmin), copy.deepcopy(mesh.Cmax)
        min_max_Cmin = np.array([np.mean(mesh.Cmin) - np.std(mesh.Cmin), np.mean(mesh.Cmin) + np.std(mesh.Cmin)])
        min_max_Cmax = np.array([np.mean(mesh.Cmax) - np.std(mesh.Cmax), np.mean(mesh.Cmax) + np.std(mesh.Cmax)])
        for i in range(mesh.Cmin.shape[0]):
            if Cmin1[i] > min_max_Cmin[1]:
                Cmin1[i] = min_max_Cmin[1]
            elif Cmin1[i] < min_max_Cmin[0]:
                Cmin1[i] = min_max_Cmin[0]
            if Cmax1[i] > min_max_Cmax[1]:
                Cmax1[i] = min_max_Cmax[1]
            elif Cmax1[i] < min_max_Cmax[0]:
                Cmax1[i] = min_max_Cmax[0]

        X[:, 0], X[:, 1] = copy.deepcopy(Cmin1[:, 0]), copy.deepcopy(Cmax1[:, 0])
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=N_clusters)
        kmeans.fit(X)
        idx_K = kmeans.predict(X)
        centers = kmeans.cluster_centers_

        # 2. Выявляем зерна (по двум кривизнам)
        
        # Сохранение данных кластеризации
        scipy.io.savemat(path_file + name_safe + '.mat', {'idx_K': idx_K,
                                                          'centers': centers})
    else:
        claster_mat = scipy.io.loadmat(path_file + name_safe + '.mat')
        idx_K, centers = np.array(claster_mat['idx_K']), np.array(claster_mat['centers'])

    # Прорисовка решения (карта кривизны вершин)
    if pl[2] == 1:
        plt.scatter(X[:, 0], X[:, 1], c=idx_K, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()

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
        sff.plot_stl_vertices_klast(struct_seg, num_segments, color_segmetns, surface_seg, mesh.vertices, idx_K,
                                    title)
    return mesh

    # Прорисовка на mesh
    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           process=False)

    vtkmeshes = trimesh2vtk(mesh)
    vtkmeshes1 = trimesh2vtk(mesh)

    # vtkmeshes.pointColors(Cmin, cmap='jet')
    vtkmeshes.pointColors(y_kmeans, cmap='jet')

    vtkmeshes.addScalarBar(title="Cmin-Cmax_k_means")
    show(vtkmeshes) #, N=1, bg='w', axes=1)
