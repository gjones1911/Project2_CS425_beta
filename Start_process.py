import numpy
import matplotlib
import matplotlib.pyplot as plt
import ML_Visualizations
import csv
import json
from Process_CSV_to_Json import load_numpy_da_file
from Process_CSV_to_Json import get_basic_stats
from Process_CSV_to_Json import get_attrib_array

file_to_find = 'UTK-peers-data_avg.dt'

data_labels, utk_avg_data, np_utk_avg = load_numpy_da_file(file_to_find, labels=True, data_delim=' ')

print(data_labels)
print('there are {:d} rows or observations'.format(len(utk_avg_data)))
print('there are {:d} cols or attributes'.format(len(utk_avg_data[0])))
print('')
obs_att = np_utk_avg.shape
n_obs = obs_att[0]
n_attrb = obs_att[1]

print('or there are {:d} observations and {:d} attributes'.format(n_obs, n_attrb))

num_obs = len(utk_avg_data)
num_vars = len(utk_avg_data[0])

attrib_array = get_attrib_array(np_utk_avg)
utk_data_stats = get_basic_stats(np_utk_avg)

#for row in utk_avg_data:
#    print(row)

u, s, vh = numpy.linalg.svd(np_utk_avg, full_matrices=True, compute_uv=True)

sum_s = sum(s.tolist())
v = numpy.transpose(vh)
s_sum = numpy.cumsum(s)
var = numpy.power(s, 2)
var2 = s**2
print('shape of s')
print(s.shape)
eigen_vals = s**2/sum_s
single_vals = numpy.arange(num_obs)

print('single_vals')
print(single_vals)

print('Variances:')
print(var.shape)
print(var.tolist())
print(var2.tolist())
#sum_s = sum(var.tolist())
print('sum of s')
print(s_sum)
print(sum_s)

print('singular values:')
print(s.tolist())
print('')
print('vh transpose')


fig = plt.figure(figsize=(8,5))
plt.plot(single_vals, eigen_vals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()