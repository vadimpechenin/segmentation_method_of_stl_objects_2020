""" Модуль вспомогательных функций для segmentation_method_of_stl_objects"""
from mpl_toolkits import mplot3d
import matplotlib.colors as colors
from matplotlib import pyplot
import numpy as np

def name_of_results(pl_sphere_cyl):
    """Функция для сохранения имени результатов этапов метода"""
    if (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==1):
        name_file='Model_Obr_ol_rus_1_t'
    elif (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==2):
        name_file='Model_Obr_ol_rus_5_t'
    elif  (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==3):
        name_file='Model_Obr_ol_rus_4_t'
    elif  (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==4):
        name_file='Model_Obr_ol_rus_2_t'
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==1):
        name_file='Model_Obr_ol_rus_1'
    elif  (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==2):
        name_file='Model_Obr_ol_rus_5'
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==3):
        name_file='Model_Obr_ol_rus_4'
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==4):
        name_file='Model_Obr_ol_rus_2'
    elif  (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==2):
        name_file='Model_Sector_3_blades_t'
    elif  (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==1):
        name_file='Model_turbine_blades_t'
    elif  (pl_sphere_cyl[0]==4)&(pl_sphere_cyl[1]==1):
        name_file='Model_turbine_blades'
    elif  (pl_sphere_cyl[0]==4)&(pl_sphere_cyl[1]==2):
        name_file='Model_Sector_3_blades'
    elif  (pl_sphere_cyl[0]==5)&(pl_sphere_cyl[1]==1):
        name_file='Three_spheres_radius64148'

    return name_file

def plot_stl_color(struct_seg,num_segments,color_segmetns,surface_seg,vertices,title):
    """Функция для прорисовки stl объекта"""
    # struct_seg - структура участков (если поверхность сегментирована, то больше 1)
    # num_segments - структура количества фасет
    # color_segmetns - структура для создания цвета участков (от 1 до n целых чисел для участков)
    # surface_seg - фасеты, из которых состоит каждый участок
    # vertices - общий набор вершин поверхности
    # title - название рисунка
    fig = pyplot.figure()
    ax = mplot3d.Axes3D(fig)
    for j in range(struct_seg.shape[0]):
        for i in range(num_segments.shape[0]):
            t=surface_seg[j][i][0]
            faces=np.zeros([t.shape[0],t.shape[1]])
            t=None
            faces=surface_seg[j][i][0]
            v0=vertices[faces[:,0]]
            v1=vertices[faces[:,1]]
            v2=vertices[faces[:,2]]
            vectors=np.zeros([v0.shape[0],3,3])
            for ij in range(v0.shape[0]):
                vectors[ij,:,:]=np.array([v0[ij,:],v1[ij,:],v2[ij,:]])
                vtx=np.array([v0[ij,:],v1[ij,:],v2[ij,:]])
                tri = mplot3d.art3d.Poly3DCollection([vtx])
                #tri.set_color(colors.rgb2hex(sp.rand(3)))
                #tri.set_edgecolor('k')
                ax.add_collection3d(tri)

    #scale = [vertices[:,0].max-vertices[:,0].min,vertices[:,1].max-vertices[:,1].min,vertices[:,2].max-vertices[:,2].min]
    #ax.auto_scale_xyz(scale, scale, scale)
    ax.set_xlim(np.amin(vertices[:][0])-2, np.amax(vertices[:][0])+2)
    ax.set_ylim(np.amin(vertices[:][1])-2, np.amax(vertices[:][1])+2)
    ax.set_zlim(np.amin(vertices[:][2])-2, np.amax(vertices[:][2])+2)
    ax.auto_scale_xyz(1, 1, 1)
    pyplot.show()