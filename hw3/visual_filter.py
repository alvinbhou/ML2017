import matplotlib.pyplot as plt

def main():
    fig = plt.figure(figsize=(14,8)) # 大小可自行決定
    for i in range(nb_filter): # 畫出每一個filter
        ax = fig.add_subplot(nb_filter/16,16,i+1) # 每16個小圖一行
        ax.imshow(image,cmap='BuGn') # image為某個filter的output或最能activate某個filter的input image
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel('whatever subfigure title you want') # 如果你想在子圖下加小標的話
        plt.tight_layout()
    fig.suptitle('Whatever title you want')
    fig.savefig(os.path.join(img_path,'Whatever filename you want')) #將圖片儲存至disk