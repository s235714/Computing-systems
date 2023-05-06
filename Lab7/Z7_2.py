import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float

def create_environment():
    context = cl.create_some_context(interactive=False)#no prompt for platform choice
    queue = cl.CommandQueue(context)
    prog = cl.Program(context, open('./imageFillIntKernel.cl').read()).build()
    return (context,queue,prog)

def convolve(image, kernel, strides: int = 1, padding=0):

    kernel = np.flipud(np.fliplr(kernel))

    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]


    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - yKernShape:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output

if __name__ == '__main__':

    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]]
    padding = 0

    context, queue, prog = create_environment()

    sampler = cl.Sampler(context,False,cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST)
    im_src = img_as_float(io.imread("Lenna.png")).astype(np.float32)
    im_dst = np.empty_like(im_src, dtype=np.float32)

    im_shape = im_src.shape[0:2]
    src_buff = cl.image_from_array(context, im_src, mode='r',num_channels=3)
    dst_buff = cl.image_from_array(context, im_dst, mode='w', num_channels=3)

    global_size = im_shape[::-1]
    local_size = None

    cl.enqueue_copy(queue, dest=im_dst, src=dst_buff, is_blocking=True, origin=(0, 0), region=im_shape[::-1])
    output = convolve(im_src, kernel, padding=1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im_src)
    ax2.imshow(output)
    plt.savefig("obraz.png",bbox_inches='tight')
