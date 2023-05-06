import pyopencl as cl

def print_dev_image_props(dev):
    properties = [
        ('Image buffer size', dev.image_max_buffer_size/1024**2),
        ('Max samplers', dev.max_samplers),
        ('Image 2D max height',dev.image2d_max_height), ('Image 2D max width',dev.image2d_max_width),
        ('Image 3D max height',dev.image3d_max_height), ('Image 3D max width',dev.image3d_max_width), ('Image 3D max depth',dev.image3d_max_depth)
    ]
    [print("{}\t:\t{}".format(name,prop)) for name, prop in properties]

if __name__ == '__main__':
    for platform in cl.get_platforms():
        for dev in platform.get_devices():
            print("\nDevice name: ", dev.name)
            if dev.image_support == 1:
                print_dev_image_props(dev)
            else:
                print("No image support")
