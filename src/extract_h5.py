import h5py
import numpy as np
import tensorflow as tf

hf = h5py.File('../data/glove/glove-100-angular.hdf5', 'r')
feat_dim = 100
batch_size = 128
topk = 100

train_x = np.array(hf.get('train')).astype(np.float32)
train_x_ = tf.transpose(tf.linalg.normalize(train_x, axis=1)[0])

test_x = np.array(hf.get('test')).astype(np.float32)
test_x_ = tf.transpose(tf.linalg.normalize(test_x, axis=1)[0])

# row_sums = np.sum(test_x, axis=1)
# test_x_ = tf.transpose(tf.constant(test_x/row_sums[:, np.newaxis], dtype=tf.float32))

distances = np.array(hf.get('distances'))
neighbors = np.array(hf.get('neighbors'))

x = tf.placeholder(tf.float32,shape=[None,feat_dim])
x_normalized = tf.linalg.normalize(x,axis=1)[0]

dist = tf.matmul(x_normalized, train_x_)

nns = tf.nn.top_k(dist, k=topk, sorted=True)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)


train_neighbors = np.zeros([train_x_.shape[1],topk],dtype=np.int32)
train_distances = np.zeros([train_x_.shape[1],topk],dtype=np.float32)


num_batches = train_x_.shape[1]//batch_size
if train_x_.shape[1]%batch_size!=0:
    num_batches += 1

for i in range(num_batches):
    start_idx = batch_size*i
    end_idx = min(start_idx+batch_size, train_x_.shape[1])
    temp = sess.run(nns, feed_dict={x: train_x[start_idx:end_idx]})
    train_neighbors[start_idx:end_idx] = temp[1]
    train_distances[start_idx:end_idx] = temp[0]

fw = open('train.txt','w')

for i in range(train_neighbors.shape[0]):
    line = ','.join([str(train_neighbors[i][j])+':'+str(train_distances[i][j]) for j in range(topk)])+' '
    line += ' '.join([str(j)+':'+str(train_x[i][j]) for j in range(feat_dim)])
    nothing = fw.write(line+'\n')

fw.close()


fw = open('test.txt','w')

for i in range(neighbors.shape[0]):
    line = ','.join([str(neighbors[i][j])+':'+str(distances[i][j]) for j in range(topk)])+' '
    line += ' '.join([str(j)+':'+str(test_x[i][j]) for j in range(feat_dim)])
    nothing = fw.write(line+'\n')

fw.close()


for n_cluster in n_cluster_l:
    print('n_cluster {}'.format(n_cluster))
    opt.n_clusters = n_cluster
    opt.n_class = n_cluster
    for height in height_l:
        run_kmkahip(height, opt, dataset, queryset, neighbors)