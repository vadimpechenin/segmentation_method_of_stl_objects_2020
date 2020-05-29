""" Модуль вспомогательных функций для segmentation_method_of_stl_objects"""
from mpl_toolkits import mplot3d
import matplotlib.colors as colors
from matplotlib import pyplot
import numpy as np
import random
#https://pypi.org/project/trimesh/
#https://trimsh.org/trimesh.visual.color.html
import trimesh
import copy
from vtkplotter import trimesh2vtk, show

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

    return N_klast

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

def plot_stl_vertices_color(struct_seg,num_segments,color_segmetns,surface_seg,vertices,Cmin,Cmax,title):
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