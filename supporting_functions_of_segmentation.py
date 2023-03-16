""" Модуль вспомогательных функций для segmentation_method_of_stl_objects"""
import matplotlib.colors as mcolors
from matplotlib import pyplot
import matplotlib.cm as cm
from scipy import interpolate
import numpy as np
import random
#https://pypi.org/project/trimesh/
#https://trimsh.org/trimesh.visual.color.html
import trimesh
import copy

#from vtkplotter import trimesh2vtk, show, Plotter
#замена vtkplotter
from vedo import show, Plotter, Points # vtkplotter
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
    elif  (pl_sphere_cyl[0]==9)&(pl_sphere_cyl[1]==1):
        name_file='SA_A10_t'
    elif (pl_sphere_cyl[0] == 10) & (pl_sphere_cyl[1] == 1):
        name_file = 'SA_A10'
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
    elif  (pl_sphere_cyl[0]==9)&(pl_sphere_cyl[1]==1):
        N_klast=30
    elif  (pl_sphere_cyl[0]==10)&(pl_sphere_cyl[1]==1):
        N_klast=30
    return N_klast

def tolerances_for_segmentation(pl_sphere_cyl,mesh):
    """Функция для сохранения допусков параметров при сегментации"""
    curveTolerance = np.std(mesh.Cmax) * 0.05
    angleTolerance = 30
    if (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==1):
        curveTolerance = np.std(mesh.Cmin) * 0.05
    elif (pl_sphere_cyl[0] == 5) & (pl_sphere_cyl[1] == 1):
        curveTolerance = 0.2
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
                meshFromTrimesh = trimesh.Trimesh(vertices=vertices,
                                       faces=faces,
                                       process=False)
                # Если объект один
                if (struct_seg.shape[0] == 1) & (num_segments.shape[0] == 1):
                    #mesh.visual.face_colors = [200, 200, 250, 100]
                    meshFromTrimesh.visual.face_colors = [200, 200, 250]
                    #mesh.visual.color.ColorVisuals(mesh=None, face_colors=[200, 200, 250], vertex_colors=None)
                else:
                    facet = range(faces.shape[0])
                    meshFromTrimesh.visual.face_colors[facet] = trimesh.visual.random_color()

        meshFromTrimesh.show()

def plot_stl_vertices_curvature(struct_seg,num_segments,color_segmetns,surface_seg,vertices,Cmin,Cmax,title):
    """Функция для прорисовки вершин stl объекта, основанных на цвете по кривизне"""
    for j in range(struct_seg.shape[0]):
        for i in range(num_segments.shape[0]):
            faces = copy.deepcopy(surface_seg[j][i][0])
            mesh = trimesh.Trimesh(vertices=vertices,
                                   faces=faces,
                                   process=False)

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

            try:
                vtkmeshes = trimesh2vtk(mesh)
                vtkmeshes1 = trimesh2vtk(mesh)
                vtkmeshes.pointColors(Cmin1, cmap='jet')

                vtkmeshes1.pointColors(Cmax1, cmap='jet')

                vtkmeshes.addScalarBar(title="Cmin")
                vtkmeshes1.addScalarBar(title="Cmax")
                show([vtkmeshes, vtkmeshes1], N=2, bg='w', axes=1)
            except:
                pc1 = Points(mesh.vertices, r=10)
                pc1.cmap("jet", Cmin1)

                pc2 = Points(mesh.vertices, r=10)
                pc2.cmap("jet", Cmax1)

                show([(mesh, pc1), (mesh, pc2)], N=2, axes=1)

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
            meshes[0].visual.face_colors = trimesh.visual.interpolate(curvature_face_klast[:,0], color_map='jet')
            meshes[1].visual.face_colors = trimesh.visual.interpolate(curvature_face_klast[:,1], color_map='jet')
            normalize = mcolors.Normalize(vmin=np.min(Cmin1), vmax=np.max(Cmin1))
            colormap = cm.jet
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(Cmin1)
            pyplot.colorbar(scalarmappaple)
            vtkmeshes = [trimesh2vtk(m) for m in meshes]
            vtkmeshes[0].addScalarBar(title="Cmin")
            vtkmeshes[1].addScalarBar(title="Cmax")
            show([vtkmeshes[0], vtkmeshes[1]], interactive=0, N=2, bg='w', axes=1) #bg2='wheat',


def plot_stl_faces_segmentation(struct_seg1,num_segments1,surface_seg1,
                                   vertices, title, noize=None):
    """Функция для прорисовки фасет stl объекта, основанных на цвете по кривизне для каждой фасеты.
       Исключается шум, заданный пользователем
       """
    struct_seg = np.array([])
    num_segments = np.array([])
    surface_seg = []
    if noize is None:
        # Если noize не задано - не надо улалять шум, оставляем структуру как есть
        struct_seg = copy.deepcopy(struct_seg1).astype('int')
        num_segments = copy.deepcopy(num_segments1)
        surface_seg = copy.deepcopy(surface_seg1)
    else:
        idx = np.where(num_segments1 > noize)
        if len(idx[0]) > 0:
            for j in range(len(idx[0])):
                surface_seg.append([])
                t = surface_seg1[idx[0][j]]
                surface_seg[j] = copy.deepcopy(t)
                num_segments = np.append(num_segments, num_segments1[idx[0][j]])
            struct_seg = np.append(struct_seg, len(idx[0])).astype('int')
        else:
            print('Все участки меньше допуска, рисуем все')
            struct_seg = copy.deepcopy(struct_seg1).astype('int')
            num_segments = copy.deepcopy(num_segments1[0, :])
            surface_seg = copy.deepcopy(surface_seg1)
    vp = Plotter(title=title, interactive=0, axes=0)
    # работа с палитрой
    lab = parula(struct_seg[0])
    for num in range(0, struct_seg[0]):
        mesh = trimesh.Trimesh(vertices=vertices,
                               faces=surface_seg[num],
                               process=False)
        mesh.visual.face_colors = lab[num, :]
        vp += mesh
    vp.show()

def plot_stl_vertices_klast(struct_seg,num_segments,color_segmetns,surface_seg,vertices,y_kmeans,title):
    """Функция для прорисовки вершин stl объекта, основанных на цвете по кривизне"""
    global pc
    for j in range(struct_seg.shape[0]):
        for i in range(num_segments.shape[0]):
            faces = copy.deepcopy(surface_seg[j][i][0])
            mesh = trimesh.Trimesh(vertices=vertices,
                                   faces=faces,
                                   process=False)
            try:
                vtkmeshes = trimesh2vtk(mesh)

                vtkmeshes.pointColors(y_kmeans, cmap='jet')

                vtkmeshes.addScalarBar(title="Cmin-Cmax_k_means")
                show(vtkmeshes)
            except:
                pc = Points(mesh.vertices, r=10)
                pc.cmap("jet", y_kmeans)
                pc.addScalarBar(title="Cmin-Cmax_k_means")
                show(pc)

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
        abcdef=np.linalg.lstsq(function_of_curve,f)
        a, b, c = abcdef[0][0], abcdef[0][1], abcdef[0][2]
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

def perimeter_and_diametr_calculation(vertices, faces):
    """Функция для расчета периметра сетки
    """
    sum1, sum2, sum3 = np.sum(((vertices[faces[:, 0], 0] - vertices[faces[:, 1], 0]) ** 2 +
                          (vertices[faces[:, 0], 1] - vertices[faces[:, 1], 1]) ** 2 +
                          (vertices[faces[:, 0], 2] - vertices[faces[:, 1], 2]) ** 2) ** 0.5),\
                    np.sum(((vertices[faces[:, 1], 0] - vertices[faces[:, 2], 0]) ** 2 +
                           (vertices[faces[:, 1], 1] - vertices[faces[:, 2], 1]) ** 2 +
                           (vertices[faces[:, 1], 2] - vertices[faces[:, 2], 2]) ** 2) ** 0.5),\
                    np.sum(((vertices[faces[:, 2], 0] - vertices[faces[:, 0], 0]) ** 2 +
                           (vertices[faces[:, 2], 1] - vertices[faces[:, 0], 1]) ** 2 +
                           (vertices[faces[:, 2], 2] - vertices[faces[:, 0], 2]) ** 2) ** 0.5)
    perimeter = np.sum(np.array([sum1, sum2, sum3]))
    deameter = np.max(np.array([(np.max(vertices[:, 0]) - np.min(vertices[:, 0])),
                                (np.max(vertices[:, 1]) - np.min(vertices[:, 1])),
                                (np.max(vertices[:, 2]) - np.min(vertices[:, 2]))]))
    return perimeter, deameter
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

def facet_exception_lite(faces_all, surface):
    """Фукция исключения значений фасет из общего массива фасет
    """
    indices1, indices2, indices3 = np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 0], surface[:, 0])], \
                                   np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 1], surface[:, 1])], \
                                   np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 2], surface[:, 2])]
    indices_ = np.intersect1d(indices1, indices2)
    indices = np.intersect1d(indices_, indices3)
    faces_all = np.delete(faces_all, indices, axis=0)
    return faces_all

def facet_intersection(faces_all, surface):
    """Фукция поиска общей матрицы фасет двух поверхностей
    """
    indices1, indices2, indices3 = np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 0], surface[:, 0])], \
                                   np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 1], surface[:, 1])], \
                                   np.arange(faces_all.shape[0])[np.in1d(faces_all[:, 2], surface[:, 2])]
    indices_ = np.intersect1d(indices1, indices2)
    indices = np.intersect1d(indices_, indices3)
    return indices

def boundary_intersection(faces_all, surface):
    """Фукция поиска общей границы двух фасетных поверхностей
    """
    indices = np.intersect1d(np.vstack((faces_all[:,0], faces_all[:,1], faces_all[:,2])),
                             np.vstack((surface[:,0], surface[:,1], surface[:,2])))
    return indices

def filling_in_common_vertices(surface_search_boundary, faces_all, boundary):
    """Функция заполнения массива общими вершинами
    """
    for i in range(faces_all.shape[1]):
        numbers, ai, ia=  np.intersect1d(faces_all[:, i], boundary, return_indices=True)
        surface_search_boundary[ai,i] = faces_all[ai, i]
    return surface_search_boundary

def search_matrix_compilation(v1, surf_sear_bound):
    """Составление матриц расстояний в массиве вершин
    """
    P_surface_search_boundary_matrix=np.zeros([surf_sear_bound.shape[0]*v1.shape[1],1])
    for i in range(v1.shape[1]):
        if i<2:
            k = i+1
        else:
            k=0
        P_surface_search_boundary_matrix[i*surf_sear_bound.shape[0]:(i+1)*surf_sear_bound.shape[0],0] = \
                                                 ((v1[surf_sear_bound[:, i], 0]-v1[surf_sear_bound[:, k], 0])**2 +
                                                 (v1[surf_sear_bound[:, i], 1]-v1[surf_sear_bound[:, k], 1])**2 +
                                                 (v1[surf_sear_bound[:, i], 2]-v1[surf_sear_bound[:, k], 2])**2)**0.5
    return P_surface_search_boundary_matrix

def creating_an_array_of_borders(surface_search,F_massiv_boundary):
    """Порядковые номера фасет границы
    """
    adfs, ai, ia = np.intersect1d(surface_search[:, 0], F_massiv_boundary, return_indices=True)
    adfs1, ai1, ia1 = np.intersect1d(surface_search[:, 1], F_massiv_boundary, return_indices=True)
    adfs2, ai2, ia2 = np.intersect1d(surface_search[:, 2], F_massiv_boundary, return_indices=True)
    array_boundary=copy.deepcopy(ai)
    array_boundary = np.append(array_boundary, ai1)
    array_boundary = np.append(array_boundary, ai2)
    array_boundary = np.unique(array_boundary)
    return array_boundary

def stl_Area_segment(p,t):
    """Функция для расчета площади сегмента stl-поверхности
    """
    d13 = np.array([(p[0, t[1, :]] - p[0, t[2, :]]), (p[1, t[1, :]] - p[1, t[2, :]]), (p[2, t[1, :]] - p[2, t[2, :]])])
    d12 = np.array([(p[0, t[0, :]] - p[0, t[1, :]]), (p[1, t[0, :]] - p[1, t[1, :]]), (p[2, t[0, :]] - p[2, t[1, :]])])
    cr = np.cross(d13.T, d12.T).T
    area = 0.5 * (cr[0, :] ** 2 + cr[1, :] ** 2 + cr[2, :] ** 2) ** (0.5)
    return np.sum(area)

def parula(length_of_array):
    """PARULA Blue-green-orange-yellow color map
        Copyright 2013-2016 The MathWorks, Inc.
    """
    matrix = np.array([
        0.2422, 0.1504, 0.6603
        , 0.2444, 0.1534, 0.6728
        , 0.2464, 0.1569, 0.6847
        , 0.2484, 0.1607, 0.6961
        , 0.2503, 0.1648, 0.7071
        , 0.2522, 0.1689, 0.7179
        , 0.2540, 0.1732, 0.7286
        , 0.2558, 0.1773, 0.7393
        , 0.2576, 0.1814, 0.7501
        , 0.2594, 0.1854, 0.7610
        , 0.2611, 0.1893, 0.7719
        , 0.2628, 0.1932, 0.7828
        , 0.2645, 0.1972, 0.7937
        , 0.2661, 0.2011, 0.8043
        , 0.2676, 0.2052, 0.8148
        , 0.2691, 0.2094, 0.8249
        , 0.2704, 0.2138, 0.8346
        , 0.2717, 0.2184, 0.8439
        , 0.2729, 0.2231, 0.8528
        , 0.2740, 0.2280, 0.8612
        , 0.2749, 0.2330, 0.8692
        , 0.2758, 0.2382, 0.8767
        , 0.2766, 0.2435, 0.8840
        , 0.2774, 0.2489, 0.8908
        , 0.2781, 0.2543, 0.8973
        , 0.2788, 0.2598, 0.9035
        , 0.2794, 0.2653, 0.9094
        , 0.2798, 0.2708, 0.9150
        , 0.2802, 0.2764, 0.9204
        , 0.2806, 0.2819, 0.9255
        , 0.2809, 0.2875, 0.9305
        , 0.2811, 0.2930, 0.9352
        , 0.2813, 0.2985, 0.9397
        , 0.2814, 0.3040, 0.9441
        , 0.2814, 0.3095, 0.9483
        , 0.2813, 0.3150, 0.9524
        , 0.2811, 0.3204, 0.9563
        , 0.2809, 0.3259, 0.9600
        , 0.2807, 0.3313, 0.9636
        , 0.2803, 0.3367, 0.9670
        , 0.2798, 0.3421, 0.9702
        , 0.2791, 0.3475, 0.9733
        , 0.2784, 0.3529, 0.9763
        , 0.2776, 0.3583, 0.9791
        , 0.2766, 0.3638, 0.9817
        , 0.2754, 0.3693, 0.9840
        , 0.2741, 0.3748, 0.9862
        , 0.2726, 0.3804, 0.9881
        , 0.2710, 0.3860, 0.9898
        , 0.2691, 0.3916, 0.9912
        , 0.2670, 0.3973, 0.9924
        , 0.2647, 0.4030, 0.9935
        , 0.2621, 0.4088, 0.9946
        , 0.2591, 0.4145, 0.9955
        , 0.2556, 0.4203, 0.9965
        , 0.2517, 0.4261, 0.9974
        , 0.2473, 0.4319, 0.9983
        , 0.2424, 0.4378, 0.9991
        , 0.2369, 0.4437, 0.9996
        , 0.2311, 0.4497, 0.9995
        , 0.2250, 0.4559, 0.9985
        , 0.2189, 0.4620, 0.9968
        , 0.2128, 0.4682, 0.9948
        , 0.2066, 0.4743, 0.9926
        , 0.2006, 0.4803, 0.9906
        , 0.1950, 0.4861, 0.9887
        , 0.1903, 0.4919, 0.9867
        , 0.1869, 0.4975, 0.9844
        , 0.1847, 0.5030, 0.9819
        , 0.1831, 0.5084, 0.9793
        , 0.1818, 0.5138, 0.9766
        , 0.1806, 0.5191, 0.9738
        , 0.1795, 0.5244, 0.9709
        , 0.1785, 0.5296, 0.9677
        , 0.1778, 0.5349, 0.9641
        , 0.1773, 0.5401, 0.9602
        , 0.1768, 0.5452, 0.9560
        , 0.1764, 0.5504, 0.9516
        , 0.1755, 0.5554, 0.9473
        , 0.1740, 0.5605, 0.9432
        , 0.1716, 0.5655, 0.9393
        , 0.1686, 0.5705, 0.9357
        , 0.1649, 0.5755, 0.9323
        , 0.1610, 0.5805, 0.9289
        , 0.1573, 0.5854, 0.9254
        , 0.1540, 0.5902, 0.9218
        , 0.1513, 0.5950, 0.9182
        , 0.1492, 0.5997, 0.9147
        , 0.1475, 0.6043, 0.9113
        , 0.1461, 0.6089, 0.9080
        , 0.1446, 0.6135, 0.9050
        , 0.1429, 0.6180, 0.9022
        , 0.1408, 0.6226, 0.8998
        , 0.1383, 0.6272, 0.8975
        , 0.1354, 0.6317, 0.8953
        , 0.1321, 0.6363, 0.8932
        , 0.1288, 0.6408, 0.8910
        , 0.1253, 0.6453, 0.8887
        , 0.1219, 0.6497, 0.8862
        , 0.1185, 0.6541, 0.8834
        , 0.1152, 0.6584, 0.8804
        , 0.1119, 0.6627, 0.8770
        , 0.1085, 0.6669, 0.8734
        , 0.1048, 0.6710, 0.8695
        , 0.1009, 0.6750, 0.8653
        , 0.0964, 0.6789, 0.8609
        , 0.0914, 0.6828, 0.8562
        , 0.0855, 0.6865, 0.8513
        , 0.0789, 0.6902, 0.8462
        , 0.0713, 0.6938, 0.8409
        , 0.0628, 0.6972, 0.8355
        , 0.0535, 0.7006, 0.8299
        , 0.0433, 0.7039, 0.8242
        , 0.0328, 0.7071, 0.8183
        , 0.0234, 0.7103, 0.8124
        , 0.0155, 0.7133, 0.8064
        , 0.0091, 0.7163, 0.8003
        , 0.0046, 0.7192, 0.7941
        , 0.0019, 0.7220, 0.7878
        , 0.0009, 0.7248, 0.7815
        , 0.0018, 0.7275, 0.7752
        , 0.0046, 0.7301, 0.7688
        , 0.0094, 0.7327, 0.7623
        , 0.0162, 0.7352, 0.7558
        , 0.0253, 0.7376, 0.7492
        , 0.0369, 0.7400, 0.7426
        , 0.0504, 0.7423, 0.7359
        , 0.0638, 0.7446, 0.7292
        , 0.0770, 0.7468, 0.7224
        , 0.0899, 0.7489, 0.7156
        , 0.1023, 0.7510, 0.7088
        , 0.1141, 0.7531, 0.7019
        , 0.1252, 0.7552, 0.6950
        , 0.1354, 0.7572, 0.6881
        , 0.1448, 0.7593, 0.6812
        , 0.1532, 0.7614, 0.6741
        , 0.1609, 0.7635, 0.6671
        , 0.1678, 0.7656, 0.6599
        , 0.1741, 0.7678, 0.6527
        , 0.1799, 0.7699, 0.6454
        , 0.1853, 0.7721, 0.6379
        , 0.1905, 0.7743, 0.6303
        , 0.1954, 0.7765, 0.6225
        , 0.2003, 0.7787, 0.6146
        , 0.2061, 0.7808, 0.6065
        , 0.2118, 0.7828, 0.5983
        , 0.2178, 0.7849, 0.5899
        , 0.2244, 0.7869, 0.5813
        , 0.2318, 0.7887, 0.5725
        , 0.2401, 0.7905, 0.5636
        , 0.2491, 0.7922, 0.5546
        , 0.2589, 0.7937, 0.5454
        , 0.2695, 0.7951, 0.5360
        , 0.2809, 0.7964, 0.5266
        , 0.2929, 0.7975, 0.5170
        , 0.3052, 0.7985, 0.5074
        , 0.3176, 0.7994, 0.4975
        , 0.3301, 0.8002, 0.4876
        , 0.3424, 0.8009, 0.4774
        , 0.3548, 0.8016, 0.4669
        , 0.3671, 0.8021, 0.4563
        , 0.3795, 0.8026, 0.4454
        , 0.3921, 0.8029, 0.4344
        , 0.4050, 0.8031, 0.4233
        , 0.4184, 0.8030, 0.4122
        , 0.4322, 0.8028, 0.4013
        , 0.4463, 0.8024, 0.3904
        , 0.4608, 0.8018, 0.3797
        , 0.4753, 0.8011, 0.3691
        , 0.4899, 0.8002, 0.3586
        , 0.5044, 0.7993, 0.3480
        , 0.5187, 0.7982, 0.3374
        , 0.5329, 0.7970, 0.3267
        , 0.5470, 0.7957, 0.3159
        , 0.5609, 0.7943, 0.3050
        , 0.5748, 0.7929, 0.2941
        , 0.5886, 0.7913, 0.2833
        , 0.6024, 0.7896, 0.2726
        , 0.6161, 0.7878, 0.2622
        , 0.6297, 0.7859, 0.2521
        , 0.6433, 0.7839, 0.2423
        , 0.6567, 0.7818, 0.2329
        , 0.6701, 0.7796, 0.2239
        , 0.6833, 0.7773, 0.2155
        , 0.6963, 0.7750, 0.2075
        , 0.7091, 0.7727, 0.1998
        , 0.7218, 0.7703, 0.1924
        , 0.7344, 0.7679, 0.1852
        , 0.7468, 0.7654, 0.1782
        , 0.7590, 0.7629, 0.1717
        , 0.7710, 0.7604, 0.1658
        , 0.7829, 0.7579, 0.1608
        , 0.7945, 0.7554, 0.1570
        , 0.8060, 0.7529, 0.1546
        , 0.8172, 0.7505, 0.1535
        , 0.8281, 0.7481, 0.1536
        , 0.8389, 0.7457, 0.1546
        , 0.8495, 0.7435, 0.1564
        , 0.8600, 0.7413, 0.1587
        , 0.8703, 0.7392, 0.1615
        , 0.8804, 0.7372, 0.1650
        , 0.8903, 0.7353, 0.1695
        , 0.9000, 0.7336, 0.1749
        , 0.9093, 0.7321, 0.1815
        , 0.9184, 0.7308, 0.1890
        , 0.9272, 0.7298, 0.1973
        , 0.9357, 0.7290, 0.2061
        , 0.9440, 0.7285, 0.2151
        , 0.9523, 0.7284, 0.2237
        , 0.9606, 0.7285, 0.2312
        , 0.9689, 0.7292, 0.2373
        , 0.9770, 0.7304, 0.2418
        , 0.9842, 0.7330, 0.2446
        , 0.9900, 0.7365, 0.2429
        , 0.9946, 0.7407, 0.2394
        , 0.9966, 0.7458, 0.2351
        , 0.9971, 0.7513, 0.2309
        , 0.9972, 0.7569, 0.2267
        , 0.9971, 0.7626, 0.2224
        , 0.9969, 0.7683, 0.2181
        , 0.9966, 0.7740, 0.2138
        , 0.9962, 0.7798, 0.2095
        , 0.9957, 0.7856, 0.2053
        , 0.9949, 0.7915, 0.2012
        , 0.9938, 0.7974, 0.1974
        , 0.9923, 0.8034, 0.1939
        , 0.9906, 0.8095, 0.1906
        , 0.9885, 0.8156, 0.1875
        , 0.9861, 0.8218, 0.1846
        , 0.9835, 0.8280, 0.1817
        , 0.9807, 0.8342, 0.1787
        , 0.9778, 0.8404, 0.1757
        , 0.9748, 0.8467, 0.1726
        , 0.9720, 0.8529, 0.1695
        , 0.9694, 0.8591, 0.1665
        , 0.9671, 0.8654, 0.1636
        , 0.9651, 0.8716, 0.1608
        , 0.9634, 0.8778, 0.1582
        , 0.9619, 0.8840, 0.1557
        , 0.9608, 0.8902, 0.1532
        , 0.9601, 0.8963, 0.1507
        , 0.9596, 0.9023, 0.1480
        , 0.9595, 0.9084, 0.1450
        , 0.9597, 0.9143, 0.1418
        , 0.9601, 0.9203, 0.1382
        , 0.9608, 0.9262, 0.1344
        , 0.9618, 0.9320, 0.1304
        , 0.9629, 0.9379, 0.1261
        , 0.9642, 0.9437, 0.1216
        , 0.9657, 0.9494, 0.1168
        , 0.9674, 0.9552, 0.1116
        , 0.9692, 0.9609, 0.1061
        , 0.9711, 0.9667, 0.1001
        , 0.9730, 0.9724, 0.0938
        , 0.9749, 0.9782, 0.0872
        , 0.9769, 0.9839, 0.0805
        ])
    matrix = matrix.reshape(256,3)
    x=np.linspace(1.0, matrix.shape[0], matrix.shape[0])
    x_int = np.linspace(1.0, matrix.shape[0], length_of_array)
    map1 = interpolate.interp1d(x, matrix[:, 0], kind='linear')
    map2 = interpolate.interp1d(x, matrix[:, 1], kind='linear')
    map3 = interpolate.interp1d(x, matrix[:, 2], kind='linear')
    map1_new, map2_new, map3_new = map1(x_int), map2(x_int), map3(x_int)
    map=np.zeros([length_of_array,3])
    map[:,0], map[:,1], map[:,2] = map1_new.T, map2_new.T, map3_new.T

    return map

