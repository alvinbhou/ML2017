import matplotlib.pyplot as plt
import numpy as np
import os

nb_filter = 32
fig = plt.figure(figsize=(14,8)) # 大小可自行決定
for i in range(nb_filter): # 畫出每一個filter
    input_img_data = np.random.random((1, 48, 48, 1)) # random noise
    image = input_img_data[0]
    ax = fig.add_subplot(nb_filter/8,8,i+1) # 每16個小圖一行
    ax.imshow(image,cmap='BuGn') # image為某個filter的output或最能activate某個filter的input image
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('whatever subfigure title you want') # 如果你想在子圖下加小標的話
    plt.tight_layout()
fig.show()
fig.suptitle('Whatever title you want')
fig.savefig('tmp.png') #將圖片儲存至disk

