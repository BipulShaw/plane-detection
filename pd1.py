# Import libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from skimage.util.shape import view_as_windows
import randomcolor
import firebass as fb

def mainfunction(dim, s_shape, s_step):
    h = int(dim)
    X = int(h)
    Y = int(h)
    Z = int(h)

    print("H = ", h)

    # initialize with all zeroes
    data = np.zeros([X,Y,Z], dtype=int)

    # randomly choose starting voxel
    x = random.randint(0, X-1)
    y = random.randint(0, Y-1)
    z = random.randint(0, Z-1)
    data[x][y][z] = 1

    # maintain a visited 2d array for keeping surface 1 vxl thick
    visited = np.zeros([X,Y], dtype=int)
    visited[x][y] = 1

    i = 0
    voxel_count = h*h

    def condition(x, y, z, dx, dy, dz):
        x = int(x)
        y = int(y)
        z = int(z)
        dx = int(dx)
        dy = int(dy)
        dz = int(dz)
        
        if(x+dx < 0 or x+dx >= X): return True
        elif(y+dy < 0 or y+dy >= Y): return True
        elif(z+dz < 0 or z+dz >= Z): return True
        elif(data[x+dx][y+dy][z+dz] == 1): return True
        elif(dz == 0 and dy == 0 and dx == 0): return True
        elif(dz != 0 and dy != 0 and dx != 0): return True
        elif(dz != 0 and dy == 0 and dx == 0): return True
        elif(visited[x+dx][y+dy] == 1): return True

        return False

    q = []
    r = []
    q.append([x, y, z])

    dxx = [0, 1, -1 , 0, 1, 1, -1, -1]
    dyy = [1, 0, 0 , -1, 1, -1, 1, -1]

    flag = True

    while(len(q) > 0):
        vox = q.pop(0)
        x = vox[0]
        y = vox[1]
        z = vox[2]
        for k in range(8):
            dx = dxx[k]
            dy = dyy[k]
            dz = 0
            
            if(dx == 0 or dy == 0): dz = np.random.choice(np.arange(-1, 2), p=[0.4, 0.2, 0.4])

            #normal voxel cond.
            var = condition(x, y, z, dx, dy, dz)
            if(var): continue
            
            ##
            data[x+dx][y+dy][z+dz] = 1
            visited[x+dx][y+dy] = 1
            q.append([x+dx, y+dy, z+dz])
            i+=1
            if(i >= int(voxel_count)):
                flag = False
                break
        if(flag == False): break
        #

    plane_surfaces = np.zeros([X, Y, Z], dtype=int)

    # plot plane surfaces
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('<--- X --->')
    # ax.set_ylabel('<--- Y --->')
    # ax.set_zlabel('<--- Z --->')

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    bl = fb.fire()
    bl.download_to_filename('./ann_model.h5')

    model = tf.keras.models.load_model('ann_model.h5')
    # model.summary()

    def pred(arr):
        prediction_data = np.empty([1, 2700], dtype=int)
        prediction_data[0] = arr

        predictions = model.predict(prediction_data)
        score = tf.nn.softmax(predictions[0])

        # class_labels = ["non-plane", "plane"]
        # print(
        #     "\nThis data most likely belongs to {} with a {:.2f}% confidence."
        #     .format(class_labels[np.argmax(score)], 100 * np.max(score))
        # )

        return [np.argmax(score), 100 * np.max(score)]

    temp_data = np.empty([X,Y,Z], dtype=object)

    for i in range(h):
        for j in range(h):
            for k in range(h):
                temp_data[i][j][k] = [i, j, k]

    # print(temp_data)

    # x, x, x shape of each window
    #(6, 6) (3, 3)
    shape = s_shape
    step_val = s_step
    # shape = 7
    # step_val = 4

    windows = view_as_windows(temp_data, (shape,shape,shape), step=step_val)
    # print(windows.shape) # 6x6x6 windows andd each window 5x5x5
    w_shape_x = windows.shape[0]
    w_shape_y = windows.shape[1]
    w_shape_z = windows.shape[2]

    list_of_planes = []
    ttl = 1

    # for i in range(h//shape):
    #     for j in range(h//shape):
    #         for k in range(h//shape):

    for i in range(w_shape_x):
        for j in range(w_shape_y):
            for k in range(w_shape_z):

                sliding_window = windows[i][j][k]
                
                newdatapts = np.zeros((2700,), dtype=int)
                x = 0

                for a in range(shape):
                    for b in range(shape):
                        for c in range(shape):
                            t = sliding_window[a][b][c]
                            if(data[t[0]][t[1]][t[2]] == 1):
                                newdatapts[x] = t[0]
                                newdatapts[x+1] = t[1]
                                newdatapts[x+2] = t[2]
                                x+=3
                
                # print(newdatapts)
                val = pred(newdatapts)
                print("//", ttl, " ##### ---> ",val[0], " ---> ", val[1])
                ttl+=1

                # mark red if plane (== 1) and p > 60%
                if(val[0] == 1 and val[1] > 60.0):
                    # for l in range(0, (shape*shape*shape*3), 3):
                    for l in range(0, 2700, 3):
                        if(data[newdatapts[l]][newdatapts[l+1]][newdatapts[l+2]] == 1):
                            plane_surfaces[newdatapts[l]][newdatapts[l+1]][newdatapts[l+2]] = 1
                
                    list_of_planes.append(plane_surfaces)
                    plane_surfaces = np.zeros([X, Y, Z], dtype=int)
                                


    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    # # plt.axis('off')
    # ax.voxels(data, facecolors='grey', alpha=0.5, edgecolors=(1,1,1,0.5))

    # for plane in list_of_planes:
    #     # color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #     color = randomcolor.RandomColor().generate()
    #     ax.voxels(plane, facecolors=color[0], alpha=0.8, edgecolors='white')

    # # plt.savefig('plot.png')

    # plt.show()
    return [list_of_planes, data]
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################