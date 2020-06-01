"""Модуль для кластеризации вершин по значениям их кривизн"""

import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy
import trimesh
from vtkplotter import trimesh2vtk, show


import matplotlib.pyplot as plt

def klastering_vetices_of_mesh_by_curvature_tensor(mesh,pl_zagr,pl,pl_sphere_cyl,path_file,curvature_tolerance):
    # Кластеризация из книги Дж. Вандер Плас "Python для сложных задач. Наука о данных и машинное обучение"
    name_safe = sff.name_of_results(pl_sphere_cyl) + '_stage_2'
    # Количество кластеров
    N_clusters=sff.num_of_klasters(pl_sphere_cyl)
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

    if (pl_zagr[2] == 1):

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=N_clusters)
        kmeans.fit(X)
        idx_K = kmeans.predict(X)
        centers = kmeans.cluster_centers_

        # 2. Выявляем зерна (по двум кривизнам)
        massiv_face_klast = np.full((mesh.faces.shape[0], 1), -1)
        curvature_null = np.array([np.amax(mesh.Cmin) + 100, np.amax(mesh.Cmax) + 100])
        curvature_face_klast = np.full((mesh.faces.shape[0], 2), curvature_null)
        curvature_mean = np.zeros([N_clusters, 2])
        for j in range(centers.shape[0]):
            w = idx_K[mesh.faces[:, 0]]
            idx_K_1, = np.where(idx_K[mesh.faces[:, 0]] == j)
            idx_K_2, = np.where(idx_K[mesh.faces[:, 1]] == j)
            idx_K_3, = np.where(idx_K[mesh.faces[:, 2]] == j)
            # 2. intersection
            C1 = np.intersect1d(idx_K_1, idx_K_2)
            C2 = np.intersect1d(idx_K_2, idx_K_3)
            C3 = np.intersect1d(C1, C2)

            if C3.shape[0] > 0:
                massiv_face_klast[C3, 0] = j
                curvature_face_klast[C3, 0], curvature_face_klast[C3, 1] = mesh.Cmin[mesh.faces[C3, 0], 0], \
                                                                           mesh.Cmax[mesh.faces[C3, 0], 0]
                curvature_mean[j, :] = np.mean(curvature_face_klast[C3, :], axis=0)
        # Выявляем границы и идентифицируем их
        id, = np.where(curvature_face_klast[:, 0] == curvature_null[0])
        # Массив кривизн смешанных граней
        f_K_1, f_K_2 = np.zeros((id.shape[0], 3)), np.zeros((id.shape[0], 3))
        f_K_1[:, 0], f_K_1[:, 1], f_K_1[:, 2] = copy.deepcopy(mesh.Cmin[mesh.faces[id, 0], 0]), \
                                                copy.deepcopy(mesh.Cmin[mesh.faces[id, 1], 0]), \
                                                copy.deepcopy(mesh.Cmin[mesh.faces[id, 2], 0])
        f_K_2[:, 0], f_K_2[:, 1], f_K_2[:, 2] = copy.deepcopy(mesh.Cmax[mesh.faces[id, 0], 0]), \
                                                copy.deepcopy(mesh.Cmax[mesh.faces[id, 1], 0]), \
                                                copy.deepcopy(mesh.Cmax[mesh.faces[id, 2], 0])
        # Нулевой массив
        f_K_1_null, f_K_2_null = np.zeros((id.shape[0], 3)), np.zeros((id.shape[0], 3))
        idx_K_1_f1, idx_K_1_f_alt1 = np.where(abs(f_K_1[:, 0]) > curvature_tolerance), \
                                     np.where(abs(f_K_1[:, 0]) <= curvature_tolerance)
        idx_K_2_f1, idx_K_2_f_alt1 = np.where(abs(f_K_1[:, 1]) > curvature_tolerance), \
                                     np.where(abs(f_K_1[:, 1]) <= curvature_tolerance)
        idx_K_3_f1, idx_K_3_f_alt1 = np.where(abs(f_K_1[:, 2]) > curvature_tolerance), \
                                     np.where(abs(f_K_1[:, 2]) <= curvature_tolerance)
        idx_K_1_f2, idx_K_1_f_alt2 = np.where(abs(f_K_2[:, 0]) > curvature_tolerance), \
                                     np.where(abs(f_K_2[:, 0]) <= curvature_tolerance)
        idx_K_2_f2, idx_K_2_f_alt2 = np.where(abs(f_K_2[:, 1]) > curvature_tolerance), \
                                     np.where(abs(f_K_2[:, 1]) <= curvature_tolerance)
        idx_K_3_f2, idx_K_3_f_alt2 = np.where(abs(f_K_2[:, 2]) > curvature_tolerance), \
                                     np.where(abs(f_K_2[:, 2]) <= curvature_tolerance)
        # Объединение union
        idx_K_1_f, idx_K_1_f_alt = np.union1d(idx_K_1_f1, idx_K_1_f2), np.union1d(idx_K_1_f_alt1, idx_K_1_f_alt2)
        idx_K_2_f, idx_K_2_f_alt = np.union1d(idx_K_2_f1, idx_K_2_f2), np.union1d(idx_K_2_f_alt1, idx_K_2_f_alt2)
        idx_K_3_f, idx_K_3_f_alt = np.union1d(idx_K_3_f1, idx_K_3_f2), np.union1d(idx_K_3_f_alt1, idx_K_3_f_alt2)

        f_K_2_null[idx_K_1_f, 0] = curvature_null[1]
        f_K_2_null[idx_K_2_f, 1] = curvature_null[1]
        f_K_2_null[idx_K_3_f, 2] = curvature_null[1]
        # Ищем где 2 острых грани, или 1, или 0
        # Делаем свертку матрицы, ищем строчки с ситуациями
        f_K_2_null_svertka = np.sum(f_K_2_null, axis=1)
        # 3 острых вершины
        id_3_versh = np.where(f_K_2_null_svertka == 3 * (curvature_null[1]))
        # 2 острых вершины
        id_2_versh = np.where(f_K_2_null_svertka == 2 * (curvature_null[1]))
        # 1 острая вершина
        id_1_versh = np.where(f_K_2_null_svertka == (curvature_null[1]))
        # 0 острых вершин
        id_0_versh = np.where(f_K_2_null_svertka == 0)
        # Поиск кривизн треугольников
        # Нулевой массив для записи кривизн не острых вершинами в фасетах с острыми вершинами
        f_K_1_null2, f_K_2_null2 = np.zeros((id.shape[0], 3)), np.zeros((id.shape[0], 3))
        # Специально когда все 3 острые вершины
        f_K_1_null3, f_K_2_null3 = np.zeros((id.shape[0], 3)), np.zeros((id.shape[0], 3))
        # Свертка значений в 1 число
        f_K_1_null_svertka, f_K_2_null_svertka = np.zeros((id.shape[0], 1)), np.zeros((id.shape[0], 1))
        # Матрица разностей для поиска кластера
        delt_kriv = np.zeros((id.shape[0], centers.shape[0]))

        # массив кривизн
        f_K_1_null2[idx_K_1_f_alt, 0] = mesh.Cmin[mesh.faces[id[idx_K_1_f_alt], 0], 0]
        f_K_1_null2[idx_K_2_f_alt, 1] = mesh.Cmin[mesh.faces[id[idx_K_2_f_alt], 1], 0]
        f_K_1_null2[idx_K_3_f_alt, 2] = mesh.Cmin[mesh.faces[id[idx_K_3_f_alt], 2], 0]
        f_K_2_null2[idx_K_1_f_alt, 0] = mesh.Cmax[mesh.faces[id[idx_K_1_f_alt], 0], 0]
        f_K_2_null2[idx_K_2_f_alt, 1] = mesh.Cmax[mesh.faces[id[idx_K_2_f_alt], 1], 0]
        f_K_2_null2[idx_K_3_f_alt, 2] = mesh.Cmax[mesh.faces[id[idx_K_3_f_alt], 2], 0]

        f_K_1_null3[idx_K_1_f, 0] = mesh.Cmin[mesh.faces[id[idx_K_1_f], 0], 0]
        f_K_1_null3[idx_K_2_f, 1] = mesh.Cmin[mesh.faces[id[idx_K_2_f], 1], 0]
        f_K_1_null3[idx_K_3_f, 2] = mesh.Cmin[mesh.faces[id[idx_K_3_f], 2], 0]
        f_K_2_null3[idx_K_1_f, 0] = mesh.Cmax[mesh.faces[id[idx_K_1_f], 0], 0]
        f_K_2_null3[idx_K_2_f, 1] = mesh.Cmax[mesh.faces[id[idx_K_2_f], 1], 0]
        f_K_2_null3[idx_K_3_f, 2] = mesh.Cmax[mesh.faces[id[idx_K_3_f], 2], 0]

        # 0 случай, 3 острые вершины (судя или по 1 или по 2 кривизне)
        if (len(id_3_versh[0]) > 0):
            f_K_1_null_svertka[id_3_versh, 0] = np.sum(f_K_1_null3[id_3_versh[0], :], axis=1) / 3
            f_K_2_null_svertka[id_3_versh, 0] = np.sum(f_K_2_null3[id_3_versh[0], :], axis=1) / 3
            for j in range(centers.shape[0]):
                delt_kriv[id_3_versh, j] = ((f_K_1_null_svertka[id_3_versh, 0] -
                                             np.full((1, len(id_3_versh[0])), centers[j, 0])) ** 2 +
                                            (f_K_1_null_svertka[id_3_versh, 0] -
                                             np.full((1, len(id_3_versh[0])), centers[j, 1])) ** 2) ** 0.5

            num_center_of_sharp = np.argmin(delt_kriv[id_3_versh[0], :], axis=1).T
            massiv_face_klast[id[id_3_versh], 0] = num_center_of_sharp.T
            curvature_face_klast[id[id_3_versh], 0] = f_K_1_null_svertka[id_3_versh, 0]
            curvature_face_klast[id[id_3_versh], 1] = f_K_2_null_svertka[id_3_versh, 0]

        # 1 случай, 2 острые вершины
        if (len(id_2_versh[0]) > 0):
            f_K_1_null_svertka[id_2_versh, 0] = np.sum(f_K_1_null3[id_2_versh[0], :], axis=1)
            f_K_2_null_svertka[id_2_versh, 0] = np.sum(f_K_2_null3[id_2_versh[0], :], axis=1)
            for j in range(centers.shape[0]):
                delt_kriv[id_2_versh, j] = ((f_K_1_null_svertka[id_2_versh, 0] -
                                             np.full((1, len(id_2_versh[0])), centers[j, 0])) ** 2 +
                                            (f_K_1_null_svertka[id_2_versh, 0] -
                                             np.full((1, len(id_2_versh[0])), centers[j, 1])) ** 2) ** 0.5
            num_center_of_sharp = np.argmin(delt_kriv[id_2_versh[0], :], axis=1).T
            massiv_face_klast[id[id_2_versh], 0] = num_center_of_sharp.T
            curvature_face_klast[id[id_2_versh], 0] = f_K_1_null_svertka[id_2_versh, 0]
            curvature_face_klast[id[id_2_versh], 1] = f_K_2_null_svertka[id_2_versh, 0]

        # 2 случай, 1 острые вершины
        if (len(id_1_versh[0]) > 0):
            f_K_1_null_svertka[id_1_versh, 0] = np.sum(f_K_1_null3[id_1_versh[0], :], axis=1)
            f_K_2_null_svertka[id_1_versh, 0] = np.sum(f_K_2_null3[id_1_versh[0], :], axis=1)
            for j in range(centers.shape[0]):
                delt_kriv[id_1_versh, j] = ((f_K_1_null_svertka[id_1_versh, 0] -
                                             np.full((1, len(id_1_versh[0])), centers[j, 0])) ** 2 +
                                            (f_K_1_null_svertka[id_1_versh, 0] -
                                             np.full((1, len(id_1_versh[0])), centers[j, 1])) ** 2) ** 0.5
            num_center_of_sharp = np.argmin(delt_kriv[id_1_versh[0], :], axis=1).T
            massiv_face_klast[id[id_1_versh], 0] = num_center_of_sharp.T
            curvature_face_klast[id[id_1_versh], 0] = f_K_1_null_svertka[id_1_versh, 0]
            curvature_face_klast[id[id_1_versh], 1] = f_K_2_null_svertka[id_1_versh, 0]

        # 3 случай, 0 острые вершины
        if (len(id_0_versh[0]) > 0):
            f_K_1_null_svertka[id_0_versh, 0] = np.sum(f_K_1_null3[id_0_versh[0], :], axis=1)
            f_K_2_null_svertka[id_0_versh, 0] = np.sum(f_K_2_null3[id_0_versh[0], :], axis=1)
            for j in range(centers.shape[0]):
                delt_kriv[id_0_versh, j] = ((f_K_1_null_svertka[id_0_versh, 0] -
                                             np.full((1, len(id_0_versh[0])), centers[j, 0])) ** 2 +
                                            (f_K_1_null_svertka[id_0_versh, 0] -
                                             np.full((1, len(id_0_versh[0])), centers[j, 1])) ** 2) ** 0.5
            num_center_of_sharp = np.argmin(delt_kriv[id_0_versh[0], :], axis=1).T
            massiv_face_klast[id[id_0_versh], 0] = num_center_of_sharp.T
            curvature_face_klast[id[id_0_versh], 0] = f_K_1_null_svertka[id_0_versh, 0]
            curvature_face_klast[id[id_0_versh], 1] = f_K_2_null_svertka[id_0_versh, 0]
        # Сохранение данных кластеризации
        scipy.io.savemat(path_file + name_safe + '.mat', {'idx_K': idx_K,
                                                          'centers': centers,
                                                          'massiv_face_klast': massiv_face_klast,
                                                          'curvature_face_klast': curvature_face_klast,
                                                          'curvature_mean': curvature_mean})
    else:
        claster_mat = scipy.io.loadmat(path_file + name_safe + '.mat')
        idx_K1, centers, massiv_face_klast, \
        curvature_face_klast, curvature_mean = claster_mat['idx_K'], \
                                               np.array(claster_mat['centers']), \
                                               np.array(claster_mat['massiv_face_klast']), \
                                               np.array(claster_mat['curvature_face_klast']), \
                                               np.array(claster_mat['curvature_mean'])
        idx_K = idx_K1[0, :].T

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

    # Вывод решения
    return idx_K, centers, massiv_face_klast, curvature_face_klast, curvature_mean