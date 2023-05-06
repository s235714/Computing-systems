import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float

def create_environment():
    context = cl.create_some_context(interactive=False)#no prompt for platform choice
    queue = cl.CommandQueue(context)
    prog = cl.Program(context, open('./Z7_0.cl').read()).build()
    return (context,queue,prog)

if __name__ == '__main__':

    context, queue, prog = create_environment()

    sampler = cl.Sampler(context,False,cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST)
    im_src = img_as_float(io.imread("Lenna.png")).astype(np.float32)
    im_dst = np.empty_like(im_src, dtype=np.float32)

    im_shape = im_src.shape[0:2]
    num_channels = 3
    src_buff = cl.image_from_array(context, im_src, mode='r',num_channels=3)
    dst_buff = cl.image_from_array(context, im_dst, mode='w', num_channels=3)

    global_size = im_shape[::-1]
    local_size = None

    prog.conv_corr(queue, global_size, local_size,sampler, src_buff, dst_buff)

    cl.enqueue_copy(queue, dest=im_dst, src=dst_buff, is_blocking=True, origin=(0, 0), region=im_shape[::-1])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im_src)
    ax2.imshow(im_dst)
    plt.savefig("obraz.png",bbox_inches='tight')
