"""Использование opencv для картинок-пояснений к отчету
"""
import cv2
import numpy as np

path_file = 'D:\\Vadim\\Семантическая сегментация нейросети\\2020\\'
path_file='results\\'
if (1==0):
    name_file1='Step_1_les_rus_5_t_py.jpg'
    name_file2='Step_1_les_rus_5_t_py_final.jpg'
else:
    name_file1='Step_1_Sector_3_blades_t_py.jpg'
    name_file2='Step_1_Sector_3_blades_t_py_final.jpg'


def save_image_rotation(img1,num_cols1,num_rows1,path_file,name_file1):
    # Изменение масштаба
    img_scaled1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Афиинные преобразования
    t = 1.0
    t1 = 0.6
    t2 = -0.6
    t4 = 0.5
    null_c, null_r = int(t4 * (num_cols1 - 1)), int(t4 * (num_rows1 - 1))
    src_points = np.float32([[null_c, null_r], [null_c + num_cols1 - 1, null_r], [null_c, null_r + num_rows1 - 1]])
    # dst_points = np.float32([[0,0], [-int(0.3*(num_cols1-1)), 0], [0,-int(0.3(num_rows1-1))]])
    # dst_points = np.float32([[0,0], [int(t*(num_cols1-1)),int(t1*(num_rows1-1))], [int(t2*(num_rows1-1)),int(t*(num_rows1-1))]])
    dst_points = np.float32([[null_c, null_r], [null_c + int(t1 * (num_cols1 - 1)), null_r + int(t2 * (num_rows1 - 1))],
                             [null_c, null_r + num_rows1 - 1]])
    affine_matrix1 = cv2.getAffineTransform(src_points, dst_points)
    img_affine1 = cv2.warpAffine(img_scaled1, affine_matrix1, (num_cols1, num_rows1))

    translation_matrix = np.float32([[1, 0, int(0.5 * num_cols1)], [0, 1, int(0.5 * num_rows1)]])

    img_output1 = cv2.warpAffine(img_affine1, translation_matrix, (2 * num_cols1, 2 * num_rows1))
    num_rows1_out, num_cols1_out = img_output1.shape[:2]
    cropped = img_output1[int(num_rows1_out * 0.2):int(num_rows1_out * 0.7),
              int(num_cols1_out * 0.2):int(num_cols1_out * 0.7)]
    cv2.imshow('Rotation', cropped)
    cv2.imwrite(path_file + 'rot_' + name_file1, cropped)
    cv2.waitKey()

#Чтение файлов
t=path_file + name_file1
img1 = cv2.imread(path_file + name_file1)
img2 = cv2.imread(path_file + name_file2)

#cv2.imshow('begin', img2)
#cv2.waitKey()
num_rows1, num_cols1 = img1.shape[:2]
num_rows2, num_cols2 = img2.shape[:2]


save_image_rotation(img1,num_cols1,num_rows1,path_file,name_file1)
save_image_rotation(img2,num_cols2,num_rows2,path_file,name_file2)


