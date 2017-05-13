import word2vec
import numpy as np
from sklearn.manifold import TSNE
import nltk, time
import matplotlib.pyplot as plt
from adjustText import adjust_text

start_time = time.time()

modelPath ='hp/all.bin'
word2vec.word2vec('hp/all.txt', modelPath, size=100, verbose=True)

model = word2vec.load(modelPath)
 
vocabs = []                 
vecs = []        

print('\n')
print(model.vocab.size )  


print('Start adding vecs')
for vocab in model.vocab:     
    vocabs.append(vocab)
    vecs.append(model[vocab])


# Get the number to plot

PLOT_NUM = 1000
    
vecs = np.array(vecs)[:PLOT_NUM]
vocabs = vocabs[:PLOT_NUM]

'''
Dimensionality Reduction
'''
# from sklearn.decomposition import PCA


print('Dimensionality Reduction')
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)


'''
Plotting
'''

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’", "</s>"]


print('filtering')
plt.figure(figsize=(18, 12))
texts = []
count = 0
for i, label in enumerate(vocabs):
    print(i, label)
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)
        count = count + 1
   

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.35))

plt.savefig( 'hp/' + str(time.time())+ 'hp' + str(PLOT_NUM) + '.png', dpi=600)

print(count)
print("--- %s seconds ---" % (time.time() - start_time))  
plt.show()