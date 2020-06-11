import supporting_functions_of_segmentation as sff
import scipy.io
import numpy as np
import copy
import trimesh

class Final_segmentation_faces():
    """Класс расчета структуры предварительной сегментации фасет"""
    def __init__(self,stl_pre_segment,mesh):
        self.pre=stl_pre_segment
        self.mesh = mesh

    def func_calculate_final_segmentation(self):
        # Функция окончательной сегментации slt и визуализации решения
        name_safe = sff.name_of_results(self.pre.pl_sphere_cyl) + '_stage_4'
        save_dict = {}  # словарь для сохранения списков с вложенными массивами
        if self.pre.pl_zagr[4] == 1:
            mesh_pre=copy.deepcopy(self.pre.mesh)
            # TODO доделать перед и после сохранения
            struct_seg=np.array([])
            surface_seg=[]
            surfaceNormal_seg=[]
            surfaceCurve_seg = []
            area_segments=np.array([])
            num_segments=np.array([])
            color_segments=np.array([])
            curve_of_segments1 = np.array([])
            for i in range(self.centers.shape[0]):
                surface=[]
                surfaceNormal=[]
                surfaceCurve=[]
                # Сохранение структуры сегментированных кусков
                surface_seg.append([])
                surface_seg[i].append(surface)
                surfaceNormal_seg.append([])
                surfaceNormal_seg[i].append(surfaceNormal)
                surfaceCurve_seg.append([])
                surfaceCurve_seg[i].append(surfaceCurve)
                # Сохранение в словарь
                save_dict['surface_seg' + str(i)] = surface_seg[i]
                save_dict['surfaceNormal_seg' + str(i)] = surfaceNormal_seg[i]
                save_dict['surfaceCurve_seg' + str(i)] = surfaceCurve_seg[i]

            struct_seg1=0
            save_dict['area_segments'] = mesh_pre.area_segments
            save_dict['num_segments'] = mesh_pre.num_segments
            save_dict['curve_of_segments'] = mesh_pre.curve_of_segments
            save_dict['color_segments'] = mesh_pre.color_segments
            save_dict['struct_seg'] = mesh_pre.struct_seg1
            scipy.io.savemat(self.pre.path_file + name_safe + '.mat', save_dict)
            struct_seg=np.append(mesh_pre.struct_seg, struct_seg1)
        else:
            segment_mat = scipy.io.loadmat(self.pre.path_file + name_safe + '.mat')
            # Извлечение массивов
            area_segments, num_segments, curve_of_segments, \
            color_segments, struct_seg1 = segment_mat['area_segments'], \
                                          segment_mat['num_segments'], \
                                          segment_mat['curve_of_segments'], \
                                          segment_mat['color_segments'], \
                                          segment_mat['struct_seg']
            struct_seg = struct_seg1[0]
            # Извлечение списков
            surface_seg = list(range(0, struct_seg[0]))
            surfaceNormal_seg = list(range(0, struct_seg[0]))
            surfaceCurve_seg = list(range(0, struct_seg[0]))
            for i in range(struct_seg[0]):
                surface_seg[i] = np.array((segment_mat['surface_seg' + str(i)])). \
                    reshape(segment_mat['surface_seg' + str(i)].shape[1],
                            segment_mat['surface_seg' + str(i)].shape[2])
                surfaceNormal_seg[i] = np.array((segment_mat['surfaceNormal_seg' + str(i)])). \
                    reshape(segment_mat['surfaceNormal_seg' + str(i)].shape[1],
                            segment_mat['surfaceNormal_seg' + str(i)].shape[2])
                surfaceCurve_seg[i] = np.array((segment_mat['surfaceCurve_seg' + str(i)])). \
                    reshape(segment_mat['surfaceCurve_seg' + str(i)].shape[1],
                            segment_mat['surfaceCurve_seg' + str(i)].shape[2])
        # Прорисовка решения (карта сегментации фасет)
        if self.pre.pl[4] == 1:
            title = 'Результат окончательной сегментации stl'
            sff.plot_stl_faces_segmentation(struct_seg,num_segments,color_segments,surface_seg,
                                        self.pre.mesh.vertices, title)

        self.mesh.surface_seg = surface_seg
        self.mesh.surfaceNormal_seg = surfaceNormal_seg
        self.mesh.surfaceCurve_seg = surfaceCurve_seg
        self.mesh.area_segments = area_segments
        self.mesh.num_segments = num_segments
        self.mesh.curve_of_segments = curve_of_segments
        self.mesh.color_segments = color_segments
        self.mesh.struct_seg = struct_seg