import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy



class Pre_segmentation_faces():
    """Класс расчета структуры предварительной сегментации фасет"""
    def __init__(self,pl_zagr,pl,pl_sphere_cyl,path_file,mesh):
        self.pl_zagr=pl_zagr
        self.pl=pl
        self.pl_sphere_cyl=pl_sphere_cyl
        self.path_file=path_file
        self.mesh = mesh
    def func_calculate_pre_segmentation(self):
        curveTolerance, angleTolerance=sff.tolerances_for_segmentation(self.pl_sphere_cyl, self.mesh)
        name_safe = sff.name_of_results(self.pl_sphere_cyl) + '_stage_3'
        if self.pl_zagr[3] == 1:

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
        if self.pl[0] == 1:
            title = 'Результат кластеризации вершин stl по кривизне'
            sff.plot_stl_vertices_klast(struct_seg, num_segments, color_segmetns, surface_seg, mesh.vertices, idx_K,
                                        title)

        return surface_seg,surfaceNormal_seg,surfaceCurve_seg,area_segments,\
               num_segments,curve_of_segments,color_segments,struct_seg