#Использование библеотеки trimesh 3.19.4
import trimesh
import tkinter.filedialog
import scipy.io
import numpy as np
#Импорт вспомогательных функций
import supporting_functions_of_segmentation as sff
#Импорт класса сетка
from mesh_class import Mesh_class
import copy


class Import_stl_data():
    """Класс для импорта структуры stl из файла"""
    def __init__(self,ex_m_var):
        self.pl_zagr=ex_m_var.pl_zagr
        self.pl=ex_m_var.pl
        self.pl_sphere_cyl=ex_m_var.pl_sphere_cyl
        self.path_file=ex_m_var.path_file
    def import_data(self):
        name_safe=sff.name_of_results(self.pl_sphere_cyl)
        if (self.pl_zagr[0]==1):
            if 1 == 1:
                file_path_string = tkinter.filedialog.askopenfilename()
                mesh_load = trimesh.load(file_path_string)
                num_vertices,num_faces =mesh_load.vertices.shape[0],mesh_load.faces.shape[0]
                vertices = copy.deepcopy(mesh_load.vertices)
                faces = copy.deepcopy(mesh_load.faces)
                normals = copy.deepcopy(mesh_load.face_normals)
                mesh = Mesh_class(num_vertices, num_faces)
                mesh.vertices =  copy.deepcopy(mesh_load.vertices)
                mesh.faces = copy.deepcopy(mesh_load.faces)
                mesh.normals = copy.deepcopy(mesh_load.face_normals)
            else:
                mesh_load_mat = scipy.io.loadmat(self.path_file + name_safe + '_matlab.mat') #+ '_matlab.mat'
                vertices = np.array(mesh_load_mat['v'])
                faces = np.array(mesh_load_mat['f'])
                for j in range(faces.shape[0]):
                    for i in range(faces.shape[1]):
                        faces[j, i] = faces[j, i] - 1
                normals = np.array(mesh_load_mat['n'])
                num_vertices, num_faces = vertices.shape[0], faces.shape[0]
                mesh = Mesh_class(num_vertices, num_faces)
                mesh.vertices = copy.deepcopy(vertices)
                mesh.faces = copy.deepcopy(faces)
                mesh.normals = copy.deepcopy(normals)
            scipy.io.savemat(self.path_file + name_safe+'.mat', {'num_vertices': num_vertices,
                                                                 'num_faces': num_faces,
                                                                 'vertices': vertices,
                                                                 'faces': faces,
                                                                 'normals': normals})
        else:
            mesh_load_mat=scipy.io.loadmat(self.path_file + name_safe+'.mat')
            mesh = Mesh_class(mesh_load_mat['num_vertices'], mesh_load_mat['num_faces'])
            mesh.vertices = np.array(mesh_load_mat['vertices'])
            mesh.faces = np.array(mesh_load_mat['faces'])
            mesh.normals = np.array(mesh_load_mat['normals'])
        if self.pl[0] == 1:
            #mesh.show()
            struct_seg=np.array([1])
            num_segments = np.array([1])
            color_segmetns=np.array([1])
            # Многомерную матрицу зададим в виде двумерного списка, у которого в ячейках будут двумерные матрицы
            surface_seg = []
            nr = 1  # количество строк
            nc = 1  # количество столбцов
            for r in range(nr):
                surface_seg.append([])
                for c in range(nc):
                    surface_seg[r].append([])
                    surface_seg[r][c].append(mesh.faces)  # добавляем очередной элемент в строку
            title='Загруженный для сегментации объект stl '
            sff.plot_stl_color(struct_seg,num_segments,color_segmetns,surface_seg,mesh.vertices,title)
        return mesh