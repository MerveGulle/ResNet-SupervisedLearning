import matplotlib.pyplot as plt


# x_CG_000, x_ResNet_000, x0_000, xref_000
for slice in range(10):
    img_name = 'xref_'+f'{slice:03d}'+'.png'
    img = plt.imread(img_name)
    img = img[13:253,96:347,:]
    figure = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    figure.savefig(img_name)