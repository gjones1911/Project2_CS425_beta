import numpy
import csv

utk_peers_csv = 'UTK-peers.csv'
ipeds_big_trimmed_csv = 'IPEDS-big-trimmed.csv'


def numpy_array_csv(csv_file):

    file_data = open(csv_file, 'rt')

    csv_reader = csv.reader(file_data, delimiter=',', quoting=csv.QUOTE_NONE)

    data_array = list(csv_reader)

    #np_data = numpy.array(data_array, dtype=numpy.float)

    attrib_list = data_array[0][1:]


    print(attrib_list)
    print(len(attrib_list))
    print(data_array[1][:])
    print(data_array[1][2:])
    #print(np_data.shape)

    file_data.close()
    return attrib_list


numpy_array_csv(utk_peers_csv)

