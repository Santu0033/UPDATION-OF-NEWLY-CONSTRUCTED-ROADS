APPLY.PY

import model

import numpy
import random
import skimage.io, skimage.morphology
import sys
import tensorflow as tf

sys.path.append('../changeseeking_tracing/')
from discoverlib import geom, graph

model_path = sys.argv[1]
in_path = sys.argv[2]
out_path = sys.argv[3]
old_tile_path = sys.argv[4]
new_tile_path = sys.argv[5]

MODEL_PATH = model_path
TEST_PATH = in_path
OUT_PATH = out_path
SAT_PATHS = [old_tile_path, new_tile_path]
VIS_PATH = None
SIZE = 1024
THRESHOLD = 0.1

print('initializing model')
m = model.Model(size=SIZE)
session = tf.Session()
m.saver.restore(session, MODEL_PATH)

print('reading inferred graph')
g = graph.read_graph(TEST_PATH)
print('creating edge index')
idx = g.edgeIndex()

# determine which tiles the graph spans
print('identifying spanned tiles')
tiles = set()
for vertex in g.vertices:
	x, y = vertex.point.x/4096, vertex.point.y/4096
	tiles.add(geom.Point(x, y))

counter = 0

out_graph = graph.Graph()
out_vertex_map = {}
def get_out_vertex(p):
	if (p.x, p.y) not in out_vertex_map:
		out_vertex_map[(p.x, p.y)] = out_graph.add_vertex(p)
	return out_vertex_map[(p.x, p.y)]

for tile in tiles:
	print('...', tile)
	tile_rect = geom.Rectangle(
		tile.scale(4096),
		tile.add(geom.Point(1, 1)).scale(4096)
	)
	tile_graph = idx.subgraph(tile_rect)
	for vertex in tile_graph.vertices:
		vertex.point = vertex.point.sub(tile_rect.start)
	if len(tile_graph.edges) == 0:
		continue

	sat1 = skimage.io.imread('{}/mass_{}_{}_sat.jpg'.format(SAT_PATHS[0], tile.x, tile.y))
	sat2 = skimage.io.imread('{}/mass_{}_{}_sat.jpg'.format(SAT_PATHS[1], tile.x, tile.y))
	origin_clip_rect = geom.Rectangle(geom.Point(0, 0), geom.Point(4096-SIZE, 4096-SIZE))

	# loop through connected components, and run our model on each component
	seen = set()
	def dfs(edge, cur):
		if edge.id in seen:
			return
		seen.add(edge.id)
		cur.append(edge.id)
		for other in edge.src.out_edges:
			dfs(other, cur)
		for other in edge.dst.out_edges:
			dfs(other, cur)
	for edge in tile_graph.edges:
		if edge.id in seen:
			continue
		cur = []
		dfs(edge, cur)
		cur_edges = [tile_graph.edges[edge_id] for edge_id in cur]

		# Prune small connected components.
		cur_length = sum([edge.segment().length() for edge in cur_edges]) / 2
		if cur_length < 60:
			continue

		subg = graph.graph_from_edges(cur_edges)
		origin = random.choice(subg.vertices).point.sub(geom.Point(SIZE/2, SIZE/2))
		origin = origin_clip_rect.clip(origin)
		im1 = sat1[origin.y:origin.y+SIZE, origin.x:origin.x+SIZE, :]
		im2 = sat2[origin.y:origin.y+SIZE, origin.x:origin.x+SIZE, :]
		im2vis = numpy.copy(im2)
		mask = numpy.zeros((SIZE, SIZE), dtype='bool')
		for edge in subg.edges:
			src = edge.src.point.sub(origin)
			dst = edge.dst.point.sub(origin)
			for p in geom.draw_line(src, dst, geom.Point(SIZE, SIZE)):
				mask[p.y, p.x] = True
				im2vis[p.y, p.x, :] = [255, 255, 0]
		mask = skimage.morphology.binary_dilation(mask, selem=skimage.morphology.disk(15))
		mask = mask.astype('uint8').reshape(SIZE, SIZE, 1)
		mask_tile = numpy.concatenate([mask, mask, mask], axis=2)
		cat_im = numpy.concatenate([im1*mask_tile, im2*mask_tile, mask], axis=2)
		output = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [cat_im],
		})[0]

		if VIS_PATH is not None:
			counter += 1
			skimage.io.imsave('{}/{}-{}-a.jpg'.format(VIS_PATH, counter, output), im1)
			skimage.io.imsave('{}/{}-{}-b.jpg'.format(VIS_PATH, counter, output), im2)
			skimage.io.imsave('{}/{}-{}-bb.jpg'.format(VIS_PATH, counter, output), im2vis)
			skimage.io.imsave('{}/{}-{}-mask-a.jpg'.format(VIS_PATH, counter, output), cat_im[:, :, 0:3])
			skimage.io.imsave('{}/{}-{}-mask-b.jpg'.format(VIS_PATH, counter, output), cat_im[:, :, 3:6])
			skimage.io.imsave('{}/{}-{}-mask.png'.format(VIS_PATH, counter, output), mask[:, :, 0]*255)

		if output < THRESHOLD:
			continue

		# integrate into out_graph
		for edge in subg.edges:
			src = edge.src.point.add(tile_rect.start)
			dst = edge.dst.point.add(tile_rect.start)
			out_graph.add_edge(get_out_vertex(src), get_out_vertex(dst))

out_graph.save(OUT_PATH)



MODEL.PY

import numpy
import tensorflow as tf
import os
import os.path
import random
import math
import time
from PIL import Image

BATCH_SIZE = 4
KERNEL_SIZE = 3

class Model:
	def _conv_layer(self, name, input_var, stride, in_channels, out_channels, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		padding = options.get('padding', 'SAME')
		batchnorm = options.get('batchnorm', False)
		transpose = options.get('transpose', False)

		with tf.variable_scope(name) as scope:
			if not transpose:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels]
			else:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, out_channels, in_channels]
			kernel = tf.get_variable(
				'weights',
				shape=filter_shape,
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / KERNEL_SIZE / KERNEL_SIZE / in_channels)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[out_channels],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			if not transpose:
				output = tf.nn.bias_add(
					tf.nn.conv2d(
						input_var,
						kernel,
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			else:
				batch = tf.shape(input_var)[0]
				side = tf.shape(input_var)[1]
				output = tf.nn.bias_add(
					tf.nn.conv2d_transpose(
						input_var,
						kernel,
						[batch, side * stride, side * stride, out_channels],
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def _fc_layer(self, name, input_var, input_size, output_size, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		batchnorm = options.get('batchnorm', False)

		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights',
				shape=[input_size, output_size],
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / input_size)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[output_size],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			output = tf.matmul(input_var, weights) + biases
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def __init__(self, bn=False, size=256):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.uint8, [None, size, size, 7])
		self.float_inputs = tf.concat([
			tf.cast(self.inputs[:, :, :, 0:6], tf.float32) / 255,
			tf.cast(self.inputs[:, :, :, 6:7], tf.float32),
		], axis=3)
		self.targets = tf.placeholder(tf.float32, [None])
		self.learning_rate = tf.placeholder(tf.float32)

		# layers
		self.layer1 = self._conv_layer('layer1', self.float_inputs, 2, 7, 64, {'batchnorm': False}) # -> 128x128x64
		self.layer2 = self._conv_layer('layer2', self.layer1, 2, 64, 128, {'batchnorm': bn}) # -> 64x64x128
		self.layer3 = self._conv_layer('layer3', self.layer2, 2, 128, 256, {'batchnorm': bn}) # -> 32x32x256
		self.layer4 = self._conv_layer('layer4', self.layer3, 2, 256, 512, {'batchnorm': bn}) # -> 16x16x512
		self.layer5 = self._conv_layer('layer5', self.layer4, 2, 512, 512, {'batchnorm': bn}) # -> 8x8x512
		self.layer6 = self._conv_layer('layer6', self.layer5, 1, 512, 512, {'batchnorm': bn}) # -> 8x8x512
		self.layer7 = self._conv_layer('layer7', self.layer6, 2, 512, 512, {'batchnorm': bn, 'transpose': True}) # -> 16x16x512
		self.layer8 = self._conv_layer('layer8', tf.concat([self.layer7, self.layer4], axis=3), 2, 1024, 256, {'batchnorm': bn, 'transpose': True}) # -> 32x32x256
		self.layer9 = self._conv_layer('layer9', tf.concat([self.layer8, self.layer3], axis=3), 2, 512, 128, {'batchnorm': bn, 'transpose': True}) # -> 64x64x128
		self.layer10 = self._conv_layer('layer10', tf.concat([self.layer9, self.layer2], axis=3), 2, 256, 64, {'batchnorm': bn, 'transpose': True}) # -> 128x128x64
		self.pre_outputs = self._conv_layer('pre_outputs', self.layer10, 2, 64, 1, {'activation': 'none', 'batchnorm': False, 'transpose': True})[:, :, :, 0] # -> 256x256x1
		#self.logit_sum = tf.reduce_sum(self.pre_outputs * self.float_inputs[:, :, :, 6], axis=[1, 2])
		#self.count_sum = tf.reduce_sum(self.float_inputs[:, :, :, 6], axis=[1, 2])
		#self.logit_avg = self.logit_sum / self.count_sum
		#self.outputs = tf.nn.sigmoid(self.logit_avg)
		#self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.logit_avg)
		self.prob_sum = tf.reduce_sum(tf.nn.sigmoid(self.pre_outputs) * self.float_inputs[:, :, :, 6], axis=[1, 2])
		self.count_sum = tf.reduce_sum(self.float_inputs[:, :, :, 6], axis=[1, 2])
		self.outputs = self.prob_sum / self.count_sum
		self.outputs_max = tf.reduce_max(tf.nn.sigmoid(self.pre_outputs) * self.float_inputs[:, :, :, 6], axis=[1, 2])
		self.targets_tile = tf.tile(tf.reshape(self.targets, [-1, 1, 1]), [1, size, size])
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_tile, logits=self.pre_outputs) * self.float_inputs[:, :, :, 6])

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)

