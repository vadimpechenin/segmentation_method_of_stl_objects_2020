""" Модуль вспомогательных функций для segmentation_method_of_stl_objects"""
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot
import matplotlib.cm as cm
from colorspacious import cspace_converter
from collections import OrderedDict
import numpy as np
import random
#https://pypi.org/project/trimesh/
#https://trimsh.org/trimesh.visual.color.html
import trimesh
import copy
from vtkplotter import trimesh2vtk, show, Plotter
import math

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
    elif  (pl_sphere_cyl[0]==6)&(pl_sphere_cyl[1]==1):
        name_file='Sphere_plane'
    return name_file

def num_of_klasters(pl_sphere_cyl):
    """Функция для сохранения количества кластеров"""
    if (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==1):
        N_klast = 9
    elif (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==2):
        N_klast = 13
    elif  (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==3):
        N_klast = 9
    elif  (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==4):
        N_klast=3
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==1):
        N_klast = 9
    elif  (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==2):
        N_klast=13
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==3):
        N_klast=9
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==4):
        N_klast = 3
    elif  (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==2):
        N_klast = 16
    elif  (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==1):
        N_klast=24
    elif  (pl_sphere_cyl[0]==4)&(pl_sphere_cyl[1]==1):
        N_klast=16
    elif  (pl_sphere_cyl[0]==4)&(pl_sphere_cyl[1]==2):
        N_klast = 16
    elif  (pl_sphere_cyl[0]==5)&(pl_sphere_cyl[1]==1):
        N_klast=4
    elif  (pl_sphere_cyl[0]==6)&(pl_sphere_cyl[1]==1):
        N_klast=2
    return N_klast

def tolerances_for_segmentation(pl_sphere_cyl,mesh):
    """Функция для сохранения допусков параметров при сегментации"""
    curveTolerance = np.std(mesh.Cmax) * 0.05
    angleTolerance = 30
    if (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==1):
        curveTolerance = np.std(mesh.Cmin) * 0.05
    elif (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==4):
        angleTolerance = 10

    return curveTolerance, angleTolerance

def plot_stl_color(struct_seg,num_segments,color_segmetns,surface_seg,vertices,title):
    # https://pydoc.net/trimesh/2.22.26/trimesh.visual/
    #https://pypi.org/project/trimesh/
    """Функция для прорисовки stl объекта"""
    # struct_seg - структура участков (если поверхность сегментирована, то больше 1)
    # num_segments - структура количества фасет
    # color_segmetns - структура для создания цвета участков (от 1 до n целых чисел для участков)
    # surface_seg - фасеты, из которых состоит каждый участок
    # vertices - общий набор вершин поверхности
    # title - название рисунка
    if (1 == 1):
        # C Использованием библеотеки trimesh
        for j in range(struct_seg.shape[0]):
            for i in range(num_segments.shape[0]):
                faces = copy.deepcopy(surface_seg[j][i][0])
                mesh = trimesh.Trimesh(vertices=vertices,
                                       faces=faces,
                                       process=False)
                # Если объект один
                if (struct_seg.shape[0] == 1) & (num_segments.shape[0] == 1):
                    #mesh.visual.face_colors = [200, 200, 250, 100]
                    mesh.visual.face_colors = [200, 200, 250]
                    #mesh.visual.color.ColorVisuals(mesh=None, face_colors=[200, 200, 250], vertex_colors=None)
                else:
                    facet = range(faces.shape[0])
                    mesh.visual.face_colors[facet] = trimesh.visual.random_color()

        mesh.show()

    if (1 == 0):
        # Попытка собрать собственное решение по визуализации
        fig = pyplot.figure()
        ax = mplot3d.Axes3D(fig)
        for j in range(struct_seg.shape[0]):
            for i in range(num_segments.shape[0]):
                faces = copy.deepcopy(surface_seg[j][i][0])
                v0 = vertices[faces[:, 0]]
                v1 = vertices[faces[:, 1]]
                v2 = vertices[faces[:, 2]]
                vectors = np.zeros([v0.shape[0], 3, 3])
                for ij in range(v0.shape[0]):
                    vectors[ij, :, :] = np.array([v0[ij, :], v1[ij, :], v2[ij, :]])
                    vtx = np.array([v0[ij, :], v1[ij, :], v2[ij, :]])
                    tri = mplot3d.art3d.Poly3DCollection([vtx])
                    # tri.set_color(colors.rgb2hex(sp.rand(3)))
                    # tri.set_edgecolor('k')
                    ax.add_collection3d(tri)

        # scale = [vertices[:,0].max-vertices[:,0].min,vertices[:,1].max-vertices[:,1].min,vertices[:,2].max-vertices[:,2].min]
        # ax.auto_scale_xyz(scale, scale, scale)
        ax.set_xlim(np.amin(vertices[:][0]) - 2, np.amax(vertices[:][0]) + 2)
        ax.set_ylim(np.amin(vertices[:][1]) - 2, np.amax(vertices[:][1]) + 2)
        ax.set_zlim(np.amin(vertices[:][2]) - 2, np.amax(vertices[:][2]) + 2)
        ax.auto_scale_xyz(1, 1, 1)
        pyplot.show()

def plot_stl_vertices_klast(struct_seg,num_segments,color_segmetns,surface_seg,vertices,Cmin,Cmax,title):
    """Функция для прорисовки вершин stl объекта, основанных на цвете по кривизне"""
    for j in range(struct_seg.shape[0]):
        for i in range(num_segments.shape[0]):
            faces = copy.deepcopy(surface_seg[j][i][0])
            mesh = trimesh.Trimesh(vertices=vertices,
                                   faces=faces,
                                   process=False)

            vtkmeshes = trimesh2vtk(mesh)
            vtkmeshes1 = trimesh2vtk(mesh)

            # vtkmeshes.pointColors(Cmin, cmap='jet')
            Cmin1, Cmax1 = copy.deepcopy(Cmin), copy.deepcopy(Cmax)
            min_max_Cmin = np.array([np.mean(Cmin) - np.std(Cmin), np.mean(Cmin) + np.std(Cmin)])
            min_max_Cmax = np.array([np.mean(Cmax) - np.std(Cmax), np.mean(Cmax) + np.std(Cmax)])
            for i in range(Cmin.shape[0]):
                if Cmin1[i] > min_max_Cmin[1]:
                    Cmin1[i] = min_max_Cmin[1]
                elif Cmin1[i] < min_max_Cmin[0]:
                    Cmin1[i] = min_max_Cmin[0]
                if Cmax1[i] > min_max_Cmax[1]:
                    Cmax1[i] = min_max_Cmax[1]
                elif Cmax1[i] < min_max_Cmax[0]:
                    Cmax1[i] = min_max_Cmax[0]

            vtkmeshes.pointColors(Cmin1, cmap='jet')

            vtkmeshes1.pointColors(Cmax1, cmap='jet')

            vtkmeshes.addScalarBar(title="Cmin")
            vtkmeshes1.addScalarBar(title="Cmax")
            show([vtkmeshes, vtkmeshes1], N=2, bg='w', axes=1)

def plot_stl_faces_color_curvature(struct_seg,num_segments,surface_seg,
                                   vertices,curvature_face_klast, title):
    """Функция для прорисовки фасет stl объекта, основанных на цвете по кривизне для каждой фасеты"""
    for j in range(struct_seg.shape[0]):
        for i in range(num_segments.shape[0]):
            faces = copy.deepcopy(surface_seg[j][i][0])
            meshes=[trimesh.Trimesh(vertices=vertices,
                                   faces=faces,
                                   process=False) for i in range(2)]

            Cmin1, Cmax1 = copy.deepcopy(curvature_face_klast[:,0]), copy.deepcopy(curvature_face_klast[:,1])
            #np.max(Cmin1)-np.min(Cmin1)
            meshes[0].visual.face_colors = trimesh.visual.interpolate(curvature_face_klast[:,0], color_map='jet')
            meshes[1].visual.face_colors = trimesh.visual.interpolate(curvature_face_klast[:,1], color_map='jet')
            # create a scene containing the mesh and colorbar
           # pyplot.pcolor(Cmin1)
            #pyplot.colorbar()
            #scene = trimesh.Scene([mesh])
            # setup the normalization and the colormap
            normalize = mcolors.Normalize(vmin=np.min(Cmin1), vmax=np.max(Cmin1))
            colormap = cm.jet
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(Cmin1)
            pyplot.colorbar(scalarmappaple)
            vtkmeshes = [trimesh2vtk(m) for m in meshes]
            vtkmeshes[0].addScalarBar(title="Cmin")
            vtkmeshes[1].addScalarBar(title="Cmax")
            #vp = Plotter(title="Cmin", interactive=0, axes=3)
            #vp +=vtkmeshes
            #vp.show(resetcam=0)
            #vp.show()
            show([vtkmeshes[0], vtkmeshes[1]], interactive=0, N=2, bg='w', axes=1) #bg2='wheat',

            # show the scene wusing
            #scene.show()
            #mesh.show()

def plot_stl_faces_segmentation(struct_seg,num_segments,color_segments,surface_seg,
                                   vertices, title):
    """Функция для прорисовки фасет stl объекта, основанных на цвете по кривизне для каждой фасеты"""
    vp = Plotter(title=title, interactive=0, axes=3)
    # работа с палитрой
    cmaps = OrderedDict()
    cmap = 'jet'
    t=struct_seg[0]
    x = np.linspace(0.0, 1.0, struct_seg[0])
    rgb = cm.get_cmap(cmap)(x)[np.newaxis, :, :3]
    lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb).reshape(struct_seg[0],3)
    if (math.fabs(np.min(lab))+np.max(lab)<255) and (np.min(lab)<0):
        lab+=math.fabs(np.min(lab))

    for num in range(0,struct_seg[0]):
        # Get RGB values for colormap and convert the colormap in
        # CAM02-UCS colorspace.  lab[0, :, 0] is the lightness.

        mesh = trimesh.Trimesh(vertices=vertices,
                               faces=surface_seg[num],
                               process=False)
        mesh.visual.face_colors = lab[num,:]
        vp += mesh
    # vp.show(resetcam=0)
    vp.show()

def plot_stl_vertices_klast(struct_seg,num_segments,color_segmetns,surface_seg,vertices,y_kmeans,title):
    """Функция для прорисовки вершин stl объекта, основанных на цвете по кривизне"""
    for j in range(struct_seg.shape[0]):
        for i in range(num_segments.shape[0]):
            faces = copy.deepcopy(surface_seg[j][i][0])
            mesh = trimesh.Trimesh(vertices=vertices,
                                   faces=faces,
                                   process=False)

            vtkmeshes = trimesh2vtk(mesh)

            vtkmeshes.pointColors(y_kmeans, cmap='jet')

            vtkmeshes.addScalarBar(title="Cmin-Cmax_k_means")
            show(vtkmeshes)

def patchnormals_double(Fa,Fb,Fc,Vx,Vy,Vz):
    # Функция вычисления составляющих нормалей в вершинах stl
    vertices = np.zeros([Vx.shape[0], 3])
    vertices[:, 0]=Vx
    vertices[:, 1]=Vy
    vertices[:, 2]=Vz
    # 1. Вычисление всех векторов граней
    e1 = vertices[Fa,:]-vertices[Fb,:]
    e2 = vertices[Fb,:]-vertices[Fc,:]
    e3 = vertices[Fc,:]-vertices[Fa,:]
    # Нормирование векторов

    e1_norm = np.array([e1[:,0] / (e1[:, 0] ** 2 + e1[:, 1] ** 2 + e1[:, 2] ** 2)** (0.5),
                       e1[:, 1] / (e1[:, 0] ** 2 + e1[:, 1] ** 2 + e1[:, 2] ** 2) ** (0.5),
                       e1[:, 2] / (e1[:, 0] ** 2 + e1[:, 1] ** 2 + e1[:, 2] ** 2) ** (0.5)]).T
    e2_norm =np.array([e2[:,0] / (e2[:, 0] ** 2 + e2[:, 1] ** 2 + e2[:, 2] ** 2)** (0.5),
                       e2[:,1] / (e2[:, 0] ** 2 + e2[:, 1] ** 2 + e2[:, 2] ** 2) ** (0.5),
                       e2[:,2] / (e2[:, 0] ** 2 + e2[:, 1] ** 2 + e2[:, 2] ** 2) ** (0.5)]).T
    e3_norm = np.array([e3[:,0] / (e3[:, 0] ** 2 + e3[:, 1] ** 2 + e3[:, 2] ** 2)** (0.5),
                       e3[:,1] / (e3[:, 0] ** 2 + e3[:, 1] ** 2 + e3[:, 2] ** 2) ** (0.5),
                       e3[:,2] / (e3[:, 0] ** 2 + e3[:, 1] ** 2 + e3[:, 2] ** 2) ** (0.5)]).T
    # 2. Расчет угла фасет, видимых из вершин
    # равносильно np.multiply(e1_norm.T,-e3_norm.T)==e1_norm.T*(-e3_norm.T)
    Angle = np.array([np.arccos(np.sum(np.multiply(e1_norm.T,-e3_norm.T),axis=0)),
                        np.arccos(np.sum(np.multiply(e2_norm.T,-e1_norm.T),axis=0)),
                      np.arccos(np.sum(np.multiply(e3_norm.T,-e2_norm.T),axis=0))]).T
    # 3. Расчет нормалей граней
    normals_of_faces=np.cross(e1,e3)
    # 4. Расчет нормалей в вершинах
    normals_in_vertices = np.zeros([Vx.shape[0], 3])
    for i in range(Fa.shape[0]):
        normals_in_vertices[Fa[i],:]=normals_in_vertices[Fa[i],:]+normals_of_faces[i,:]*Angle[i, 0]
        normals_in_vertices[Fb[i],:]=normals_in_vertices[Fb[i],:]+normals_of_faces[i,:]*Angle[i, 1]
        normals_in_vertices[Fc[i],:]=normals_in_vertices[Fc[i],:]+normals_of_faces[i,:]*Angle[i, 2]
    eps=1/10000000000000000
    V_norm = (normals_in_vertices[:, 0] ** 2 + normals_in_vertices[:, 1] ** 2 +
              normals_in_vertices[:, 2] ** 2) ** (0.5) + eps
    Nx = normals_in_vertices[:, 0] / V_norm
    Ny = normals_in_vertices[:, 1] / V_norm
    Nz = normals_in_vertices[:, 2] / V_norm
    return Nx, Ny, Nz

def patchnormals(mesh):
    # Функция вычисления нормалей триангуляционной сетки.
    # Вызывается функция patchnormals_double, вычисляющая нормали всех граней, а затем нормали вершин по нормалям
    # граней, взвешенным по углам граней
    # t=mesh.faces[:, 0]
    # k=t.astype('float32')
    Nx, Ny, Nz = patchnormals_double(mesh.faces[:, 0], mesh.faces[:, 1],
                                     mesh.faces[:, 2], mesh.vertices[:, 0].astype('float32'),
                                     mesh.vertices[:, 1].astype('float32'), mesh.vertices[:, 2].astype('float32'))
    normals_in_vertices = np.zeros([Nx.shape[0], 3])
    normals_in_vertices[:, 0]=Nx
    normals_in_vertices[:, 1]=Ny
    normals_in_vertices[:, 2]=Nz
    return normals_in_vertices

def VectorRotationMatrix(normals_in_vertices):
    # Функция расчета матрицы вращения и обратной ей
    v = (normals_in_vertices.T) / (np.sum(normals_in_vertices**2))**(0.5)
    k = np.array([random.random() for i in range(3)])
    l = np.array([k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]])
    l = l / (np.sum(l ** 2))**(0.5)
    k = np.array([l[1] * v[2] - l[2] * v[1], l[2] * v[0] - l[0] * v[2], l[0] * v[1] - l[1] * v[0]])
    k = k / (np.sum(k ** 2))**(0.5)
    Minv = np.array([v, l, k]).T
    M = np.linalg.inv(Minv)
    return M, Minv

def vertex_neighbours_double(Fa,Fb,Fc,Vx,Vy,Vz):
    faces = np.array([Fa, Fb, Fc]).T
    vertex= np.array([Vx, Vy, Vz]).T
    # массив соседей вершин
    neighbours_of_vetres = []
    for r in range(vertex.shape[0]):
        neighbours_of_vetres.append([])
    # Пройтись по всем фасетам
    for i in range(faces.shape[0]):
        # Добавить соседей каждой вершины фасеты в список соседей.
        neighbours_of_vetres[faces[i, 0]].append(faces[i, 1])
        neighbours_of_vetres[faces[i, 0]].append(faces[i, 2])
        neighbours_of_vetres[faces[i, 1]].append(faces[i, 2])
        neighbours_of_vetres[faces[i, 1]].append(faces[i, 0])
        neighbours_of_vetres[faces[i, 2]].append(faces[i, 0])
        neighbours_of_vetres[faces[i, 2]].append(faces[i, 1])
    # Перебрать все соседние массивы и отсортировать их (Поворот такой же, как у фасет)
    for i in range(vertex.shape[0]):
        Pneighf=neighbours_of_vetres[i]
        Pneig = []
        if (len(Pneighf)==0):
           pass
        else:
            for x in Pneighf:
                if x not in Pneig:
                    Pneig.append(x)
        neighbours_of_vetres[i] = Pneig
    return neighbours_of_vetres

def vertex_neighbours(mesh):
    # Эта функция производит поиск всех фасет-соседей каждой вершины в списке фасет поверхности.
    neighbours_of_vetres = vertex_neighbours_double(mesh.faces[:, 0], mesh.faces[:, 1],
                                     mesh.faces[:, 2], mesh.vertices[:, 0],
                                     mesh.vertices[:, 1], mesh.vertices[:, 2])
    return neighbours_of_vetres

def eig2(Dxx, Dxy, Dyy):
    # | Dxx  Dxy |
    # |          |
    # | Dxy  Dyy |
    # % Пример,
    # %   Dxx=round(rand*10); Dxy=round(rand*10); Dyy=round(rand*10);
    # %   [a,b,c,d]=eig2(Dxx,Dxy,Dyy)
    # %   D = [a 0;0 b];
    # %   V = [c(:) d(:)];
    # %   check =  sum(abs(M*V(:,1) - D(1,1)*V(:,1))) + sum(abs(M*V(:,2) - D(2,2)*V(:,2))) ;
    # Вычисление собственных векторов
    tmp = ((Dxx - Dyy)** 2 + 4 * Dxy**2)**(0.5)
    v2x, v2y = 2 * Dxy, Dyy - Dxx + tmp
    # Нормализация
    mag = (v2x** 2 + v2y** 2)**(0.5)
    if (mag != 0):
        v2x = v2x / mag
        v2y = v2y / mag
    # Собственные векторы ортогональны
    v1x = -copy.deepcopy(v2y)
    v1y = copy.deepcopy(v2x)
    # Вычисление собственных значений
    mu1 = (0.5 * (Dxx + Dyy + tmp))
    mu2 = (0.5 * (Dxx + Dyy - tmp))
    #  Сортировка собственных значений по абсолютному значению abs (Lambda1) < abs(Lambda2)
    if (abs(mu1) < abs(mu2)):
        min_curvature = mu1
        max_curvature = mu2
        I2 = np.array([v1x, v1y])
        I1 = np.array([v2x, v2y])
    else:
        min_curvature = mu2
        max_curvature = mu1
        I2 = np.array([v2x, v2y])
        I1 = np.array([v1x, v1y])
    return min_curvature, max_curvature, I1, I2

def patchcurvature_2014(mesh):
    # Функция расчета кривизн, переписанная с кода
    # D.Kroon University of Twente (August 2011)

    #1. Количество вершин
    num_vertices=mesh.vertices.shape[0]
    # 2. Расчет нормалей в вершинах
    normals_in_vertices = patchnormals(mesh)
    # 3. Расчет матриц вращения для нормалей
    M = np.zeros([3, 3, num_vertices])
    Minv = np.zeros([3, 3, num_vertices])
    for i in range(num_vertices):
        M[:,:, i], Minv[:,:, i]=VectorRotationMatrix(normals_in_vertices[i,:])

    # 4. Получить соседей всех вершин
    neighbours_of_vetres=vertex_neighbours(mesh)
    # Перебрать все вершины
    min_curvature = np.zeros([num_vertices, 1])
    max_curvature = np.zeros([num_vertices, 1])
    Dir1 = np.zeros([num_vertices, 3])
    Dir2 = np.zeros([num_vertices, 3])
    for i in range(num_vertices):
        # Получение соседей первого и второго уровней.
        idx_1_2_ring=np.array(neighbours_of_vetres[i])
        neighbours_of_vetres_1_2_ring=np.array([]).astype(int)
        for j in idx_1_2_ring:
            c=np.array(neighbours_of_vetres[int(j)])
            neighbours_of_vetres_1_2_ring = np.append(neighbours_of_vetres_1_2_ring,c)
        neighbours_of_vetres_1_2_ring=np.unique(neighbours_of_vetres_1_2_ring)
        vert_no_rot=mesh.vertices[neighbours_of_vetres_1_2_ring,:]
        # Вращение вершин, чтобы сделать нормаль [-1 0 0]
        c=Minv[:,:,i]
        vert_rot=np.dot(vert_no_rot, Minv[:,:,i])
        f , x, y = vert_rot[:, 0], vert_rot[:, 1], vert_rot[:, 2]
        # Вписать фасеты в уравнение второй степени
        # f(x,y) = ax^2 + by^2 + cxy + dx + ey + f
        function_of_curve = np.array([x[:]** 2, y[:]** 2, x[:]*y[:], x[:], y[:], np.ones([vert_rot.shape[0]])]).T
        #abcdef=np.zeros([1,function_of_curve.shape[1]])
        abcdef=np.linalg.lstsq(function_of_curve,f)
        a, b, c = abcdef[0][0], abcdef[0][1], abcdef[0][2]
        #for k in range(function_of_curve.shape[1]):
        #   abcdef[:,k]=function_of_curve[:,k]/f
        # Создание матрицы Гессиана
        # H =  [2*a c;c 2*b]
        Dxx, Dxy, Dyy  = 2 * a, c, 2 * b
        min_curvature[i], max_curvature[i], I1, I2 = eig2(Dxx, Dxy, Dyy)
        dir1 = np.dot(np.array([0, I1[0], I1[1]]), M[:, :, i])
        dir2 = np.dot(np.array([0, I2[0], I2[1]]), M[:, :, i])
        Dir1[i,:]=dir1 / (dir1[0] ** 2 + dir1[1] ** 2 + dir1[2] ** 2) ** (0.5)
        Dir2[i,:]=dir2 / (dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2) ** (0.5)

    Cmean = (min_curvature + max_curvature) / 2
    Cgaussian = min_curvature * max_curvature
    return Dir1, Dir2, min_curvature, max_curvature, Cmean, Cgaussian

def ExtractSurface_by_curve_2020_2_curves(nodes, faces, normals, curves, targetPoint,
                                          targetVector,angleTolerance,pl_extract):
    """Функция для выбора группы фасет, связанных с целевым по критериям главных кривизн"""
    # 1. поиск узла, ближайшего к интересующей точке и вектора отсечения граней
    nodeIndex =node_grain(nodes,targetPoint)


    #Фильтр граней по вектору отсечения
    angles = ((curves[:, 0] - np.full((curves.shape[0],1),targetVector[0])[:,0])** 2 +
              (curves[:, 1] - np.full((curves.shape[0],1),targetVector[1])[:,0])** 2)**(0.5)
    indexes = np.where(angles < angleTolerance)
    faces = faces[indexes[0],:]
    normals = normals[indexes[0],:]
    curves = curves[indexes[0],:]
    if (1==0):
        gridsize = (1, 2)
        fig = pyplot.figure(figsize=(12, 8))
        ax2 = pyplot.subplot2grid(gridsize, (0, 0))
        ax3 = pyplot.subplot2grid(gridsize, (0, 1))
        ax2.plot(range(curves.shape[0]), curves[:, 0])
        ax3.plot(range(curves.shape[0]), curves[:, 1])
        pyplot.show()
    # грани, содержащие целевую точку
    handledFacesIndexes = np.unique(np.hstack(((np.where(faces[:, 0] == nodeIndex)[0]),
                                               np.where(faces[:, 1] == nodeIndex)[0],
                                               np.where(faces[:, 2] == nodeIndex)[0])))
    resultFaces = faces[handledFacesIndexes, :]
    resultNormals = normals[handledFacesIndexes, :]
    resultCurves = curves[handledFacesIndexes, :]

    resultFaces, resultNormals, resultCurves, handledFacesIndexes = common_part_for_functionsExtractSurface(
                                                                    resultFaces, resultNormals,
                                                                    resultCurves, handledFacesIndexes,
                                                                    nodes, faces, normals, curves,pl_extract
                                                                    )

    return resultFaces, resultNormals, resultCurves

def ExtractSurface_by_curve_2020_2_curves_norm(nodes, faces, normals, curves, targetPoint,
                                               targetVector,angleTolerance,targetVector2,angleTolerance2,pl_extract):
    """Функция для выбора группы фасет, связанных с целевым по критериям главных кривизн"""
    # 1. поиск узла, ближайшего к интересующей точке и вектора отсечения граней
    nodeIndex =node_grain(nodes,targetPoint)


    #Фильтр граней по вектору отсечения
    # Углы нормалей
    q=normals*np.full((normals.shape[0],3), targetVector2)
    t2=np.sum((normals* np.full((normals.shape[0],3), targetVector2)),axis=1)
    angles2 = np.arccos(np.sum((normals* np.full((normals.shape[0],3), targetVector2)),axis=1))/math.pi*180
    indexes2 = np.where(angles2<angleTolerance2)
    angles = ((curves[:, 0] - np.full((curves.shape[0],1),targetVector[0])[:,0])** 2 +
              (curves[:, 1] - np.full((curves.shape[0],1),targetVector[1])[:,0])** 2)**(0.5)
    indexes1 = np.where(angles < angleTolerance)
    indexes = np.intersect1d(indexes1, indexes2)
    faces = faces[indexes,:]
    normals = normals[indexes,:]
    curves = curves[indexes,:]
    # грани, содержащие целевую точку
    handledFacesIndexes = np.unique(np.hstack(((np.where(faces[:, 0] == nodeIndex)[0]),
                                               np.where(faces[:, 1] == nodeIndex)[0],
                                               np.where(faces[:, 2] == nodeIndex)[0])))
    resultFaces = faces[handledFacesIndexes, :]
    resultNormals = normals[handledFacesIndexes, :]
    resultCurves = curves[handledFacesIndexes, :]

    resultFaces, resultNormals, resultCurves, handledFacesIndexes = common_part_for_functionsExtractSurface(
                                                                        resultFaces,resultNormals,
                                                                        resultCurves,handledFacesIndexes,
                                                                        nodes,faces,normals,curves,pl_extract
                                                                    )

    return resultFaces, resultNormals, resultCurves


def node_grain(nodes,targetPoint):
    d = nodes - np.full((nodes.shape[0], 3), targetPoint)
    d = np.array([(np.sum((d * d) ** 2, axis=1)) ** 0.5]).T
    # Stacking the two arrays horizontally
    k = np.array([np.arange(0, d.shape[0], 1)]).T
    d = np.hstack((d, k))
    d = d[d[:, 0].argsort()]
    nodeIndex = d[0, 1].astype('int')
    return nodeIndex

def common_part_for_functionsExtractSurface(resultFaces,resultNormals,resultCurves,handledFacesIndexes,
                                            nodes,faces,normals,curves,pl_extract):

    # построение связей узлов с гранями. Первый столбец номер узла, второй массив с номерами граней
    g=(np.array([faces[:, 0]]).T, np.array([np.arange(0, faces.shape[0], 1)]).T)
    h=np.hstack((np.array([faces[:, 0]]).T, np.array([np.arange(0, faces.shape[0], 1)]).T))
    links = np.vstack((np.hstack((np.array([faces[:, 0]]).T, np.array([np.arange(0, faces.shape[0], 1)]).T)),
                       np.hstack((np.array([faces[:, 1]]).T, np.array([np.arange(0, faces.shape[0], 1)]).T)),
                       np.hstack((np.array([faces[:, 2]]).T, np.array([np.arange(0, faces.shape[0], 1)]).T))))
    links = links[links[:, 0].argsort()]

    links = np.vstack(([-1, - 1], links, [-1, - 1]))
    limits = np.where((links[1:, 0] - links[0: -1, 0]) != 0)[0]
    nodesFaces = []
    for r in range(nodes.shape[0]):
        nodesFaces.append([])
        for c in range(2):
            nodesFaces[r].append([])
            if (c == 0):
                nodesFaces[r][c].append(r)

    for limitIndex in range(1, len(limits)):
        nodeIndex = links[limits[limitIndex], 0]
        left = limits[limitIndex - 1] + 1
        right = limits[limitIndex]
        facesIndexes = np.sort(links[left:right+1, 1])
        nodesFaces[nodeIndex][1] = facesIndexes

    if pl_extract == 1:
        # прорисовка шагов
        vp = Plotter(title="Cmin", interactive=0, axes=3)
    while 1:
        nodeIndexes = faces[handledFacesIndexes, :]
        nodeIndexes = np.unique(nodeIndexes[:])
        intermediate_array_of_index = np.array([])
        for j in range(len(nodeIndexes)):
            intermediate_array_of_index = np.hstack((intermediate_array_of_index, nodesFaces[nodeIndexes[j]][1])).astype('int')

        checkedFaceIndexes = np.setdiff1d(np.unique(intermediate_array_of_index), handledFacesIndexes)
        if len(checkedFaceIndexes) == 0:
            break

        resultFaces = np.vstack((resultFaces, faces[checkedFaceIndexes, :]))
        resultNormals = np.vstack((resultNormals, normals[checkedFaceIndexes, :]))
        resultCurves = np.vstack((resultCurves, curves[checkedFaceIndexes, :]))
        handledFacesIndexes = np.union1d(handledFacesIndexes, checkedFaceIndexes)
        if pl_extract == 1:
            mesh = trimesh.Trimesh(vertices=nodes,
                                   faces=faces[checkedFaceIndexes, :],
                                   process=False)
            mesh.visual.face_colors = trimesh.visual.random_color()
            vp +=mesh
            #vp.show(resetcam=0)
            vp.show()
        g=0
    return resultFaces, resultNormals, resultCurves,handledFacesIndexes



def facet_exception(massiv_face_klast_all, faces_all, normals_all, curvature_face_klast_all, surface):
    """Фукция исключения сегментированных значений из массивов
    """
    indices1, indices2, indices3 = np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 0], surface[:, 0])], \
                                   np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 1], surface[:, 1])], \
                                   np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 2], surface[:, 2])]
    indices_ = np.intersect1d(indices1, indices2)
    indices = np.intersect1d(indices_, indices3)
    faces_all = np.delete(faces_all, indices, axis=0)
    normals_all=np.delete(normals_all, indices, axis=0)
    massiv_face_klast_all = np.delete(massiv_face_klast_all, indices, axis=0)
    curvature_face_klast_all = np.delete(curvature_face_klast_all, indices, axis=0)
    return massiv_face_klast_all, faces_all, normals_all, curvature_face_klast_all

def stl_Area_segment(p,t):
    """Функция для расчета площади сегмента stl-поверхности
    """
    d13 = np.array([(p[0, t[1, :]] - p[0, t[2, :]]), (p[1, t[1, :]] - p[1, t[2, :]]), (p[2, t[1, :]] - p[2, t[2, :]])])
    d12 = np.array([(p[0, t[0, :]] - p[0, t[1, :]]), (p[1, t[0, :]] - p[1, t[1, :]]), (p[2, t[0, :]] - p[2, t[1, :]])])
    cr = np.cross(d13.T, d12.T).T
    area = 0.5 * (cr[0, :] ** 2 + cr[1, :] ** 2 + cr[2, :] ** 2) ** (0.5)
    return np.sum(area)



