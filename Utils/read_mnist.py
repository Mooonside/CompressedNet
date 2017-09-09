import struct
import numpy as np

def read_image(filename):
    f = open(filename, 'rb')
    buf = f.read()
    f.close()
    # '>IIII' :using big endian,MSB stored in lower position,read four integers
    index = 0
    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    # print 'magic',magic,'images',images,'rows',rows,'cols',columns

    # struct.calcsize(I) returns the size of an integer(in bytes)
    index += struct.calcsize('>IIII')

    data_set = np.zeros([images,rows,columns,1])

    for i in xrange(images):
        im = struct.unpack_from('>784B', buf, index)
        data_set[i,:,:,:] = np.asarray(im).reshape([rows,columns,1])
        index += struct.calcsize('>784B')
    return data_set


def read_label(filename):
    f = open(filename, 'rb')

    buf = f.read()
    f.close()
    # '>IIII' :using big endian,MSB stored in lower position,read four integers
    index = 0
    magic, labels = struct.unpack_from('>II', buf, index)

    #print magic,labels

    # struct.calcsize(I) returns the size of an integer(in bytes)
    index += struct.calcsize('>II')

    data_set = np.ones(labels)

    for i in xrange(labels):
        data_set[i] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    return data_set