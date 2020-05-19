#Использование библеотеки trimesh 3.5.22
import trimesh
import tkinter.filedialog
import scipy.io
import numpy as np
#Импорт вспомогательных функций
import supporting_functions_of_segmentation as sff
#Импорт класса сетка
from mesh_class import Mesh_class

class Import_stl_data():
    """Класс для импорта структуры stl из файла"""
    def __init__(self,pl_zagr,pl,pl_sphere_cyl,path_file):
        self.pl_zagr=pl_zagr
        self.pl=pl
        self.pl_sphere_cyl=pl_sphere_cyl
        self.path_file=path_file
    def import_data(self):
        name_safe=sff.name_of_results(self.pl_sphere_cyl)
        if (self.pl_zagr[0]==1):
            file_path_string = tkinter.filedialog.askopenfilename()
            mesh_load = trimesh.load(file_path_string)
            num_vertices,num_faces =mesh_load.vertices.shape[0],mesh_load.faces.shape[0]
            mesh = Mesh_class(num_vertices, num_faces)
            vertices = mesh_load.vertices
            faces =  mesh_load.faces
            normals = mesh_load.face_normals
            mesh.vertices =  mesh_load.vertices
            mesh.faces = mesh_load.faces
            mesh.normals = mesh_load.face_normals
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
            pass
        return mesh