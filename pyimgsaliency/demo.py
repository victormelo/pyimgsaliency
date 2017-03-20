import pyimgsaliency as psal
import cv2
import os
import os.path as P
import matplotlib.pyplot as plt
# path to the image
filename = '/home/victor/clipped_live_dice_nodice/clipped_live_dice/dice-00000238-4.png'
# filename = '/home/victor/clipped_live_dice_nodice/clipped_live_dice/dice-00000000-1.png'

# get the saliency maps using the 3 implemented methods
# rbd = psal.get_saliency_rbd(filename).astype('uint8')

# ft = psal.get_saliency_ft(filename).astype('uint8')
# binary_sal = psal.binarise_saliency_map(rbd*255,method='adaptive')

# cv2.imshow('rbd',rbd)
# import pdb; pdb.set_trace()
# cv2.imshow('ft',binary_sal)

# cv2.waitKey(0)
# mbd = psal.get_saliency_mbd(filename).astype('uint8')

PATH = '/home/victor/clipped_live_dice_nodice/clipped_live_dice/'
fns = os.listdir(PATH)
for fn in [file for file in fns if file.endswith('.png')]:
    outfn = 'result-' + P.basename(fn)
    print('making: %s' % outfn)
    im = cv2.imread(P.join(PATH, fn))
    mbd = psal.get_saliency_mbd(im).astype('uint8')
    # often, it is desirable to have a binary saliency map
    binary_sal = psal.binarise_saliency_map(mbd,method='adaptive')
    plt.subplot(1,2,1), plt.imshow(im)
    plt.title('Input image')

    plt.subplot(1,2,2), plt.imshow(binary_sal)
    plt.title('Mask')

    plt.savefig('/home/victor/clipped_live_dice_nodice/out/%s' % outfn)


# # cv2.imshow('img',img)
# # cv2.imshow('rbd',rbd)
# # cv2.imshow('ft',ft)
# # cv2.imshow('mbd',mbd)

# #openCV cannot display numpy type 0, so convert to uint8 and scale
# cv2.imshow('binary',255 * binary_sal.astype('uint8'))


# cv2.waitKey(0)
