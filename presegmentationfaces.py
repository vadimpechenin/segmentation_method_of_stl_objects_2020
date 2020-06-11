import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy
import trimesh



class Pre_segmentation_faces():
    """Класс расчета структуры предварительной сегментации фасет"""
    def __init__(self,pl_zagr,pl,pl_sphere_cyl,path_file,mesh,centers,massiv_face_klast,curvature_face_klast):
        self.pl_zagr=pl_zagr
        self.pl=pl
        self.pl_sphere_cyl=pl_sphere_cyl
        self.path_file=path_file
        self.mesh = mesh
        self.centers = centers
        self.massiv_face_klast = massiv_face_klast
        self.curvature_face_klast = curvature_face_klast
    def func_calculate_pre_segmentation(self):
        # Функция предварительной сегментации slt и визуализации решения
        name_safe = sff.name_of_results(self.pl_sphere_cyl) + '_stage_3'
        save_dict={} #словарь для сохранения списков с вложенными массивами
        if self.pl_zagr[3] == 1:
            curveTolerance, angleTolerance = sff.tolerances_for_segmentation(self.pl_sphere_cyl, self.mesh)
            massiv_face_klast_all=copy.deepcopy(self.massiv_face_klast)
            faces_all = copy.deepcopy(self.mesh.faces)
            normals_all = copy.deepcopy(self.mesh.normals)
            curvature_face_klast_all = copy.deepcopy(self.curvature_face_klast)
            ko = 1 # Запись для цвета
            i = -1
            struct_seg=np.array([])
            surface_seg=[]
            surfaceNormal_seg=[]
            surfaceCurve_seg = []
            area_segments=np.array([])
            num_segments=np.array([])
            color_segments=np.array([])
            curve_of_segments1 = np.array([])
            # 3.1 Основной цикл сегментирования данных по значению кривизны
            t=()
            for j in range(self.centers.shape[0]):
                # Центральная точка
                if  (np.size(massiv_face_klast_all)>0):
                    id = np.where((massiv_face_klast_all[:, 0]) == j)
                    length_of_num_segments=len(id[0])
                while (length_of_num_segments>0):
                    i += 1
                    #Подход случайного выбора
                    if (1==0):
                        msize = len(id[0])
                        idx = np.random.permutation(msize)
                        firstpart = id[0][idx[0]]
                    else:
                        #Подход близости к центру кластера (центральному зерну)
                        array_for_first=((curvature_face_klast_all[id, 0]-
                                          np.full((len(id[0]),1),self.centers[j, 0])) ** 2 +
                                         (curvature_face_klast_all[id, 1] -
                                          np.full((len(id[0]), 1),self.centers[j, 1])) ** 2) **(0.5)
                        min_idx = np.argmin(array_for_first)
                        firstpart=id[0][min_idx]
                    targetPoint = self.mesh.vertices[faces_all[firstpart, 0],:]
                    targetVector = curvature_face_klast_all[firstpart,:]
                    targetVector2 = normals_all[firstpart,:]
                    if (abs(curvature_face_klast_all[firstpart, 0]) > 0.005) or \
                        ((abs(curvature_face_klast_all[firstpart, 1]) > 0.005)):
                        surface, surfaceNormal,surfaceCurve = sff.ExtractSurface_by_curve_2020_2_curves(
                                                                self.mesh.vertices, faces_all,
                                                                normals_all, curvature_face_klast_all,
                                                                targetPoint, targetVector, curveTolerance,0
                                                                )
                    else:
                        surface, surfaceNormal, surfaceCurve = sff.ExtractSurface_by_curve_2020_2_curves_norm(
                                                                self.mesh.vertices, faces_all,
                                                                normals_all, curvature_face_klast_all,
                                                                targetPoint, targetVector, curveTolerance,
                                                                targetVector2,angleTolerance,0
                                                                )


                    # Исключение сегментированных значений из общего массива
                    massiv_face_klast_all, faces_all, \
                    normals_all, curvature_face_klast_all=sff.facet_exception(
                                                            massiv_face_klast_all, faces_all, normals_all,
                                                            curvature_face_klast_all, surface
                                                            )
                    if massiv_face_klast_all.shape[0]>0:
                        id=np.where((massiv_face_klast_all[:, 0]) == j)
                        length_of_num_segments = len(id[0])
                    else:
                        id=()
                        length_of_num_segments = len(id)
                    # Сохранение структуры сегментированных кусков
                    surface_seg.append([])
                    surface_seg[i].append(surface)
                    surfaceNormal_seg.append([])
                    surfaceNormal_seg[i].append(surfaceNormal)
                    surfaceCurve_seg.append([])
                    surfaceCurve_seg[i].append(surfaceCurve)
                    #Сохранение в словарь
                    save_dict['surface_seg' + str(i)] = surface_seg[i]
                    save_dict['surfaceNormal_seg' + str(i)] = surfaceNormal_seg[i]
                    save_dict['surfaceCurve_seg' + str(i)] = surfaceCurve_seg[i]
                    #Вычисление площадей
                    area=sff.stl_Area_segment(self.mesh.vertices.T,surface.T)
                    area_segments = np.append(area_segments, area)
                    # Количество фасет в сегменте
                    num_segments = np.append(num_segments, surface.shape[0])
                    # Порядковый номер для цвета сегмента
                    color_segments=np.append(color_segments, ko)
                    # Кривизна сегмента
                    curve_of_segments1=np.append(curve_of_segments1, targetVector)
                    ko=ko+1
                    print('Осталось фасет фактически: '+ str(massiv_face_klast_all.shape[0])+
                          '; распознано: '+str(np.sum(num_segments))+'; осталось фасет расчетно:'+
                          str(self.mesh.faces.shape[0]-np.sum(num_segments)))
            if ko > 1:
                struct_seg1 = num_segments.shape[0]
                curve_of_segments = np.zeros([num_segments.shape[0],2])
                for numerate in range(num_segments.shape[0]):
                    curve_of_segments[numerate,:] = curve_of_segments1[0+numerate*2:1+numerate*2]
            else:
                struct_seg1 = 0
                curve_of_segments=np.array([])
            save_dict['area_segments'] = area_segments
            save_dict['num_segments'] = num_segments
            save_dict['curve_of_segments'] = curve_of_segments
            save_dict['color_segments'] = color_segments
            save_dict['struct_seg'] = struct_seg1
            scipy.io.savemat(self.path_file + name_safe + '.mat', save_dict)
            struct_seg=np.append(struct_seg, struct_seg1)
        else:
            segment_mat = scipy.io.loadmat(self.path_file + name_safe + '.mat')
            # Извлечение массивов
            area_segments, num_segments, curve_of_segments, \
            color_segments, struct_seg1 = segment_mat['area_segments'], \
                             segment_mat['num_segments'], \
                             segment_mat['curve_of_segments'], \
                             segment_mat['color_segments'],\
                             segment_mat['struct_seg']
            struct_seg=struct_seg1[0]
            # Извлечение списков
            surface_seg = list(range(0,struct_seg[0]))
            surfaceNormal_seg = list(range(0, struct_seg[0]))
            surfaceCurve_seg = list(range(0, struct_seg[0]))
            for i in range(struct_seg[0]):
                surface_seg[i] = np.array((segment_mat['surface_seg' + str(i)])).\
                    reshape(segment_mat['surface_seg' + str(i)].shape[1],
                            segment_mat['surface_seg' + str(i)].shape[2])
                surfaceNormal_seg[i] = np.array((segment_mat['surfaceNormal_seg' + str(i)])).\
                    reshape(segment_mat['surfaceNormal_seg' + str(i)].shape[1],
                            segment_mat['surfaceNormal_seg' + str(i)].shape[2])
                surfaceCurve_seg[i] = np.array((segment_mat['surfaceCurve_seg' + str(i)])).\
                    reshape(segment_mat['surfaceCurve_seg' + str(i)].shape[1],
                            segment_mat['surfaceCurve_seg' + str(i)].shape[2])
        # Прорисовка решения (карта сегментации фасет)
        if self.pl[3] == 1:
            title = 'Результат предварительной сегментации stl'
            sff.plot_stl_faces_segmentation(struct_seg,num_segments,color_segments,surface_seg,
                                        self.mesh.vertices, title)

        self.mesh.surface_seg=surface_seg
        self.mesh.surfaceNormal_seg = surfaceNormal_seg
        self.mesh.surfaceCurve_seg = surfaceCurve_seg
        self.mesh.area_segments = area_segments
        self.mesh.num_segments = num_segments
        self.mesh.curve_of_segments = curve_of_segments
        self.mesh.color_segments = color_segments
        self.mesh.struct_seg = struct_seg
