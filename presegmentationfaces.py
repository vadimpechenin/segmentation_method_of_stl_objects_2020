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

        name_safe = sff.name_of_results(self.pl_sphere_cyl) + '_stage_3'
        if self.pl_zagr[3] == 1:
            curveTolerance, angleTolerance = sff.tolerances_for_segmentation(self.pl_sphere_cyl, self.mesh)
            massiv_face_klast_all=copy.deepcopy(self.massiv_face_klast)
            faces_all = copy.deepcopy(self.mesh.faces)
            normals_all = copy.deepcopy(self.mesh.normals)
            curvature_face_klast_all = copy.deepcopy(self.curvature_face_klast)
            ko = 1 # Запись для цвета
            i = 0
            # 3.1 Основной цикл сегментирования данных по значению кривизны
            for j in range(self.centers.shape[0]):
                # Центральная точка
                if  (np.size(massiv_face_klast_all)>0):
                    id = np.where((massiv_face_klast_all[:, 0]) == j)
                while (len(id[0])>0):
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
                                                                targetPoint, targetVector, curveTolerance
                                                                )
                    else:
                        surface, surfaceNormal, surfaceCurve = sff.ExtractSurface_by_curve_2020_2_curves_norm(
                            self.mesh.vertices, faces_all,
                            normals_all, curvature_face_klast_all,
                            targetPoint, targetVector, curveTolerance,targetVector2,angleTolerance
                        )
                    mesh1 = trimesh.Trimesh(vertices=self.mesh.vertices,
                                           faces=surface,
                                           process=False)
                    mesh1.visual.face_colors = [200, 200, 250]
                    mesh1.show()
                    g=0
            scipy.io.savemat(self.path_file + name_safe + '.mat', {'surface_seg': surface_seg,
                                                                   'surfaceNormal_seg': surfaceNormal_seg,
                                                                   'surfaceCurve_seg': surfaceCurve_seg,
                                                                   'area_segments': area_segments,
                                                                   'num_segments': num_segments,
                                                                   'Curve_of_segments': curve_of_segments,
                                                                   'color_segments': color_segments,
                                                                   'struct_seg': struct_seg})
        else:
            segment_mat = scipy.io.loadmat(self.path_file + name_safe + '.mat')
            surface_seg, surfaceNormal_seg, surfaceCurve_seg, \
            area_segments, num_segments,  curve_of_segments,\
            color_segments, struct_seg = segment_mat['surface_seg'], \
                                         segment_mat['surfaceNormal_seg'], \
                                         segment_mat['surfaceCurve_seg'], \
                                         segment_mat['area_segments'], \
                                         segment_mat['num_segments'],\
                                         segment_mat['curve_of_segments'],\
                                         segment_mat['color_segments'],\
                                         segment_mat['struct_seg']


        # Прорисовка решения (карта сегментации фасет)
        if self.pl[3] == 1:
            title = 'Результат кластеризации вершин stl по кривизне'
            sff.plot_stl_vertices_klast(struct_seg, num_segments, color_segmetns, surface_seg, mesh.vertices, idx_K,
                                        title)

        return surface_seg,surfaceNormal_seg,surfaceCurve_seg,area_segments,\
               num_segments,curve_of_segments,color_segments,struct_seg