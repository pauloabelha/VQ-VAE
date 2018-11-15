import io_image
import numpy as np
import visualize as vis

img_filepath = '/home/paulo/reconstruction_train_ycb_batch_adver_10.png'
img_res = (200, 200)
n_rows = 3

def read_image_from_image_grid(img_filepath, img_res, grid_x, grid_y):
    img_grid = io_image.read_RGB_image((img_filepath))

    img = np.zeros(img_res)
    for i in range(3):
        start_y = grid_x*img_res[0]
        start_x = grid_y*img_res[1]
        img += img_grid[start_x:start_x+img_res[0], start_y:start_y+img_res[1], i] * np.power(255, i)

    img = img / 255
    return img

def read_image_column_from_image_grid(img_filepath, img_res, column, n_rows):
    imgs = []
    for i in range(n_rows):
        imgs.append(read_image_from_image_grid(img_filepath, img_res, i, column))
    return imgs

images = read_image_column_from_image_grid(img_filepath, img_res, 4, n_rows)

for i in range(3):
    vis.plot_image(images[i])

vis.show()
