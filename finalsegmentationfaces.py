import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy
import trimesh

class Final_segmentation_faces():
    """Класс расчета структуры предварительной сегментации фасет"""
    def __init__(self,stl_pre_segment,mesh):
        self.pre = stl_pre_segment
        self.mesh = mesh

    def func_calculate_final_segmentation(self):
        # Функция окончательной сегментации slt и визуализации решения
        name_safe = sff.name_of_results(self.pre.pl_sphere_cyl) + '_stage_4'
        save_dict = {}  # словарь для сохранения списков с вложенными массивами
        if self.pre.pl_zagr[4] == 1:
            mesh_pre = copy.deepcopy(self.pre.mesh)
            curveTolerance, angleTolerance = sff.tolerances_for_segmentation(self.pre.pl_sphere_cyl, mesh_pre)
            # Критерий максимальной площади
            # Общая площадь всех сегментов
            Area_of_segments_total = sff.stl_Area_segment(mesh_pre.vertices.T, mesh_pre.faces.T)
            nm = np.round(self.pre.massiv_face_klast.shape[0] / mesh_pre.struct_seg[0] * 0.2)
            eps = 1E-5
            Toleranse_area = Area_of_segments_total / self.pre.massiv_face_klast.shape[0] * nm
            Tolerance_of_curve_in_merging = curveTolerance
            # Общий периметр
            faces_perimetr, deameter_max = sff.perimeter_and_diametr_calculation(mesh_pre.vertices, mesh_pre.faces)
            # Создание матрицы координат вершин сетки, последняя точка - удаленная от сетки
            vertices1=np.zeros([mesh_pre.vertices.shape[0]+1,3])
            vertices1=np.vstack((mesh_pre.vertices, np.array([np.mean(mesh_pre.vertices,axis=0) + deameter_max * 1E+3])))
            # Находится максимальный элемент
            struct_seg1=copy.deepcopy(mesh_pre.struct_seg)
            area_segments1 = copy.deepcopy(mesh_pre.area_segments)
            num_segm_counter=copy.deepcopy(mesh_pre.struct_seg[0])
            while (num_segm_counter>1):
                max_index_Area = np.argmax(area_segments1)
                surface_search = copy.deepcopy(mesh_pre.surface_seg[max_index_Area])
                counter = 1
                while (counter == 1):
                    counter = 0
                    num_cor=np.arange(mesh_pre.struct_seg[0])
                    num_cor=np.delete(num_cor, num_cor[max_index_Area])
                    print(str(num_segm_counter))
                    for idx_of_segm in range(mesh_pre.struct_seg[0] - 1):
                        # Поиск общей границы



                        if (area_segments1[0,num_cor[idx_of_segm]] > 0):
                            surface_search2=copy.deepcopy(mesh_pre.surface_seg[num_cor[idx_of_segm]])
                            surface_search2_uniq =np.unique(np.vstack((np.array([surface_search2[:, 0]]),
                                                                      np.array([surface_search2[:, 1]]),
                                                                      np.array([surface_search2[:, 2]]))),
                                                           return_index=False, return_inverse=False,
                                                           return_counts=False, axis=None)
                            surface_search1_uniq = np.unique(np.vstack((np.array([surface_search[:, 0]]),
                                                                      np.array([surface_search[:, 1]]),
                                                                      np.array([surface_search[:, 2]]))),
                                                           return_index=False, return_inverse=False,
                                                           return_counts=False, axis=None)
                            F_massiv_boundary = np.intersect1d(surface_search1_uniq, surface_search2_uniq)
                            if (F_massiv_boundary.shape[0]) > 2:
                                # средние кривизны границы
                                c_mean_boundaru = np.array([np.mean(self.pre.curvature_face_klast[F_massiv_boundary,0]),
                                                            np.mean(self.pre.curvature_face_klast[F_massiv_boundary, 1])])
                                # Расстояние
                                DC=np.abs(((mesh_pre.curve_of_segments[max_index_Area,0]-c_mean_boundaru[0])**2+
                                           (mesh_pre.curve_of_segments[max_index_Area,1]-c_mean_boundaru[1])**2)**0.5-
                                          ((mesh_pre.curve_of_segments[num_cor[idx_of_segm],0]-c_mean_boundaru[0])**2+
                                           (mesh_pre.curve_of_segments[num_cor[idx_of_segm],1]-c_mean_boundaru[1])**2)**0.5)

                                if ((area_segments1[0, num_cor[idx_of_segm]] < Toleranse_area) or
                                        (area_segments1[0, max_index_Area] < Toleranse_area)) or \
                                        ((area_segments1[0, num_cor[idx_of_segm]] < Toleranse_area) and
                                        (area_segments1[0, max_index_Area] < Toleranse_area)):
                                    D = DC * eps
                                else:
                                    D = copy.deepcopy(DC)
                                # Поиск периметров сегментов и общей границы
                                # 2. Делаем ответные области фасету (из общей вычитаем текущие)
                                f_surface_search_sympathetic, \
                                f_surface_search_sympathetic1= copy.deepcopy(mesh_pre.faces), \
                                copy.deepcopy(mesh_pre.faces)
                                f_surface_search_sympathetic = sff.facet_exception_lite(f_surface_search_sympathetic,
                                                                                      surface_search)
                                f_surface_search_sympathetic1 = sff.facet_exception_lite(f_surface_search_sympathetic1,
                                                                                        surface_search2)
                                # 3.  Поиск пересекающихся вершин с общей структурой, т.е. границ
                                boundary_surface_poisk = sff.boundary_intersection(surface_search,
                                                                               f_surface_search_sympathetic)
                                boundary_surface_poisk1 = sff.boundary_intersection(surface_search2,
                                                                               f_surface_search_sympathetic)
                                # 4. Матрицы с v1(end+1,:) вершиной
                                surface_search_boundary = np.full((surface_search.shape[0], 3), vertices1.shape[0]-1)
                                surface_search_boundary1 = np.full((surface_search2.shape[0], 3), vertices1.shape[0]-1)
                                surface_search_boundary_common = np.full((surface_search2.shape[0], 3), vertices1.shape[0]-1)
                                # 5. Включение в матрицы вершины не 0
                                # Большая область
                                surface_search_boundary = sff.filling_in_common_vertices(surface_search_boundary,
                                                                                         surface_search,
                                                                                     boundary_surface_poisk)
                                surface_search_boundary1 = sff.filling_in_common_vertices(surface_search_boundary1,
                                                                                         surface_search2,
                                                                                         boundary_surface_poisk1)
                                surface_search_boundary_common = sff.filling_in_common_vertices(surface_search_boundary_common,
                                                                                         surface_search2,
                                                                                         F_massiv_boundary)
                                # 6. Матрицы расстояний в массиве vertices1
                                P_surface_search_boundary_matrix = sff.search_matrix_compilation(vertices1,
                                                                                        surface_search_boundary)
                                P_surface_search_boundary_matrix1 = sff.search_matrix_compilation(vertices1,
                                                                                        surface_search_boundary1)
                                P_surface_search_boundary_matrix_common = sff.search_matrix_compilation(vertices1,
                                                                                        surface_search_boundary_common)
                                # 7. Обнуление ненужных расстояний и
                                # поиск периметров
                                idx = np.where(P_surface_search_boundary_matrix[:,0] > deameter_max)
                                P_surface_search_boundary_matrix[idx,0] = 0
                                P_surface_search= np.sum(P_surface_search_boundary_matrix)
                                idx = np.where(P_surface_search_boundary_matrix1[:,0] > deameter_max)
                                P_surface_search_boundary_matrix1[idx, 0] = 0
                                P_surface_search1 = np.sum(P_surface_search_boundary_matrix1)
                                idx = np.where(P_surface_search_boundary_matrix_common[:,0] > deameter_max)
                                P_surface_search_boundary_matrix_common[idx, 0] = 0
                                P_boundary = np.sum(P_surface_search_boundary_matrix_common)
                                if P_boundary>0:
                                    D *= np.min([P_surface_search, P_surface_search1]) / P_boundary
                                else:
                                    D *= np.min([P_surface_search, P_surface_search1])

                                if abs(mesh_pre.curve_of_segments[max_index_Area,0]) < 0.005 or \
                                    abs(mesh_pre.curve_of_segments[max_index_Area,1]) < 0.005:
                                    #Массивы границ
                                    if (idx_of_segm==109):
                                        g=0
                                    array_boundary=sff.creating_an_array_of_borders(surface_search,F_massiv_boundary)
                                    norm_boundary=copy.deepcopy(mesh_pre.surfaceNormal_seg[max_index_Area][array_boundary,:])
                                    array_boundary1 = sff.creating_an_array_of_borders(surface_search2, F_massiv_boundary)
                                    norm_boundary_symp = copy.deepcopy(mesh_pre.surfaceNormal_seg[num_cor[idx_of_segm]][array_boundary1,:])
                                    norm_boundary1=np.zeros([1,3])
                                    norm_boundary_symp1 = np.zeros([1, 3])
                                    if len(array_boundary) > 1:
                                        for i in range(3):
                                            norm_boundary1[0,i] = np.mean(norm_boundary[:,i])
                                    else:
                                        norm_boundary1=copy.deepcopy(norm_boundary)
                                    if len(array_boundary1) > 1:
                                        for i in range(3):
                                            norm_boundary_symp1[0,i] = np.mean(norm_boundary_symp[:,i])
                                    else:
                                        norm_boundary_symp1 = copy.deepcopy(norm_boundary_symp)
                                    angles2 = np.arccos(np.sum((norm_boundary1* np.full((norm_boundary1.shape[0],3),
                                                                norm_boundary_symp1)),axis=1))/np.pi*180
                                    if (angles2 > angleTolerance):
                                        D = Tolerance_of_curve_in_merging * 2

                                # Объединение в случае удовлетворения условиям
                                if (D < Tolerance_of_curve_in_merging):
                                    # Поглощение сегмента
                                    surface_search = np.vstack((surface_search, surface_search2))
                                    mesh_pre.surface_seg[max_index_Area] = copy.deepcopy(surface_search)
                                    # Обнуление структуры поглощенного сегмента
                                    struct_seg1 -= 1
                                    # Структура фасет поверхности
                                    #del  mesh_pre.surface_seg[num_cor[idx_of_segm]]
                                    mesh_pre.surface_seg[num_cor[idx_of_segm]] = np.array([])
                                    # Нормали фасет поверхности
                                    normal_search = np.vstack((mesh_pre.surfaceNormal_seg[max_index_Area],
                                                               mesh_pre.surfaceNormal_seg[num_cor[idx_of_segm]]))
                                    mesh_pre.surfaceNormal_seg[max_index_Area] = copy.deepcopy(normal_search)
                                    #np.append(mesh_pre.surfaceNormal_seg[max_index_Area],
                                    #    mesh_pre.surfaceNormal_seg[num_cor[idx_of_segm]])

                                    mesh_pre.surfaceNormal_seg[num_cor[idx_of_segm]] = np.array([])
                                    # Кривизны фасет поверхностей
                                    curve_search = np.vstack((mesh_pre.surfaceCurve_seg[max_index_Area],
                                                               mesh_pre.surfaceCurve_seg[num_cor[idx_of_segm]]))
                                    mesh_pre.surfaceCurve_seg[max_index_Area] = copy.deepcopy(curve_search)
                                    #np.append(mesh_pre.surfaceCurve_seg[max_index_Area],
                                    #    mesh_pre.surfaceCurve_seg[num_cor[idx_of_segm]])
                                    mesh_pre.surfaceCurve_seg[num_cor[idx_of_segm]] = np.array([])
                                    # Площадь поверхности
                                    area_segments1[0,num_cor[idx_of_segm]] = 0
                                    # Количество элементов в сегменте
                                    mesh_pre.num_segments[max_index_Area] += mesh_pre.num_segments[num_cor[idx_of_segm]]
                                    mesh_pre.num_segments[num_cor[idx_of_segm]] = 0
                                    # Цвет сегмента убираем
                                    mesh_pre.color_segments[0,num_cor[idx_of_segm]] = 0
                                    # Кривизна сегмента
                                    mesh_pre.curve_of_segments[num_cor[idx_of_segm], 0] = 0
                                    mesh_pre.curve_of_segments[num_cor[idx_of_segm], 1] = 0
                                    num_cor[idx_of_segm]=-1
                                    num_segm_counter -= 1
                                    if counter==0:
                                        counter = 1
                        else:
                            pass
                area_segments1[0,max_index_Area] = 0
                num_segm_counter -= 1
            # Сохраниние структуры данных
            struct_seg=np.array([])
            surface_seg=[]
            surfaceNormal_seg=[]
            surfaceCurve_seg = []
            area_segments=np.array([])
            num_segments=np.array([])
            color_segments=np.array([])
            curve_of_segments = np.array([])
            idx = np.where(mesh_pre.num_segments[:] > 0)
            if len(idx[0]) > 0:
                for j in range(len(idx[0])):
                    surface_seg.append([])
                    surface = mesh_pre.surface_seg[idx[0][j]]
                    surface_seg[j] = copy.deepcopy(surface)
                    surfaceNormal_seg.append([])
                    surfaceNormal = mesh_pre.surfaceNormal_seg[idx[0][j]]
                    surfaceNormal_seg[j] = copy.deepcopy(surfaceNormal)
                    surfaceCurve_seg.append([])
                    surfaceCurve = mesh_pre.surfaceCurve_seg[idx[0][j]]
                    surfaceCurve_seg[j] = copy.deepcopy(surfaceCurve)
                    #Вычисление площадей
                    area=sff.stl_Area_segment(mesh_pre.vertices.T,surface.T)
                    area_segments = np.append(area_segments, area)
                    # Количество фасет в сегменте
                    num_segments = np.append(num_segments, mesh_pre.num_segments[idx[0][j]])
                    # Порядковый номер для цвета сегмента
                    color_segments = np.append(color_segments, j+1)
                    # Кривизна сегмента
                    curve_of_segments = np.append(curve_of_segments, mesh_pre.curve_of_segments[idx[0][j]])
                    # Сохранение в словарь
                    save_dict['surface_seg' + str(j)] = surface_seg[j]
                    save_dict['surfaceNormal_seg' + str(j)] = surfaceNormal_seg[j]
                    save_dict['surfaceCurve_seg' + str(j)] = surfaceCurve_seg[j]
            struct_seg1=num_segments.shape[0]
            save_dict['area_segments'] = area_segments
            save_dict['num_segments'] = num_segments
            save_dict['curve_of_segments'] = curve_of_segments
            save_dict['color_segments'] = color_segments
            save_dict['struct_seg'] = struct_seg1
            scipy.io.savemat(self.pre.path_file + name_safe + '.mat', save_dict)
            struct_seg=np.append(struct_seg, struct_seg1)
        else:
            segment_mat = scipy.io.loadmat(self.pre.path_file + name_safe + '.mat')
            # Извлечение массивов
            area_segments, num_segments1, curve_of_segments, \
            color_segments, struct_seg1 = segment_mat['area_segments'], \
                                          segment_mat['num_segments'], \
                                          segment_mat['curve_of_segments'], \
                                          segment_mat['color_segments'], \
                                          segment_mat['struct_seg']
            num_segments = copy.deepcopy(num_segments1[0, :])
            struct_seg = struct_seg1[0]
            # Извлечение списков
            surface_seg = list(range(0, struct_seg[0]))
            surfaceNormal_seg = list(range(0, struct_seg[0]))
            surfaceCurve_seg = list(range(0, struct_seg[0]))
            for i in range(struct_seg[0]):
                surface_seg[i] = np.array((segment_mat['surface_seg' + str(i)]))#. \
                    #reshape(segment_mat['surface_seg' + str(i)].shape[1],
                    #        segment_mat['surface_seg' + str(i)].shape[2])
                surfaceNormal_seg[i] = np.array((segment_mat['surfaceNormal_seg' + str(i)]))#. \
                    #reshape(segment_mat['surfaceNormal_seg' + str(i)].shape[1],
                    #        segment_mat['surfaceNormal_seg' + str(i)].shape[2])
                surfaceCurve_seg[i] = np.array((segment_mat['surfaceCurve_seg' + str(i)]))#. \
                    #reshape(segment_mat['surfaceCurve_seg' + str(i)].shape[1],
                    #        segment_mat['surfaceCurve_seg' + str(i)].shape[2])
        # Прорисовка решения (карта сегментации фасет)
        if self.pre.pl[4] == 1:
            title = 'Результат окончательной сегментации stl'
            sff.plot_stl_faces_segmentation(struct_seg,num_segments,surface_seg,
                                        self.mesh.vertices, title)

        self.mesh.surface_seg = surface_seg
        self.mesh.surfaceNormal_seg = surfaceNormal_seg
        self.mesh.surfaceCurve_seg = surfaceCurve_seg
        self.mesh.area_segments = area_segments
        self.mesh.num_segments = num_segments
        self.mesh.curve_of_segments = curve_of_segments
        self.mesh.color_segments = color_segments
        self.mesh.struct_seg = struct_seg