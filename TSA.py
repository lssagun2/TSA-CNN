!pip install sigfig
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import random
import time
import sigfig as sf
num_epochs = 10
header1 = f'{"".center(8)}|{"Results".center(25 + num_epochs*13)}|{"C1 Layer".center(37)}|{"C3 Layer".center(37)}|{"F5 Layer".center(27)}|{"F6 Layer".center(27)}|{"General".center(22)}|'
header2 = f'{"Tree".center(8)}|{"Accuracy".center(12)}|'
for i in range(num_epochs):
    add = "Epoch " + str(i+1)
    header2 += f'{add.center(12)}|'
header2 += f'{"Running".center(12)}|{"Number of".center(13)}{"Kernel".center(10)}{"Activation".center(14)}|{"Number of".center(13)}{"Kernel".center(10)}{"Activation".center(14)}|{"Number of".center(13)}{"Activation".center(14)}|{"Number of".center(13)}{"Activation".center(14)}|{"Batch".center(9)}{"Optimizer".center(13)}|'
header3 = f'{"".center(8)}|{"".center(12)}|'
for i in range(num_epochs):
    header3 += f'{"".center(12)}|'
header3 += f'{"time".center(12)}|{"filters".center(13)}{"size".center(10)}{"function".center(14)}|{"filters".center(13)}{"size".center(10)}{"function".center(14)}|{"neurons".center(13)}{"function".center(14)}|{"neurons".center(13)}{"function".center(14)}|{"size".center(9)}{"".center(13)}|'

def chooseBest(solutions, f, train_set, test_set, epochs, filename, verbose = False):
	# results = [f(t, epochs) for t in solutions]
	results = []
	for i in range(len(solutions)):
		hyperparameters = solutions[i]
		result = f(hyperparameters, train_set, test_set, epochs)
		if verbose:
			displayTree(i+1, hyperparameters, *result, filename)
		results.append(result)
	values, times, val_history = zip(*results)
	values, times, val_history = list(values), list(times), list(val_history)
	best_val = max(values)
	best_index = values.index(best_val)
	return values, times, val_history, best_index

def initializeTrees(population_size, dimensions, search_space):
	trees = []
	for i in range(population_size):
		tree = [0 for d in range(dimensions)]
		for d in range(dimensions):
			space_min, space_max = search_space[d]
			tree[d] = space_min + random.random() * (space_max - space_min)
		trees.append(tree)
	return trees

def initializeSeeds(trees, current_tree_index, best_tree_index, population_size, search_tendency, dimensions, search_space):
	num_seeds = int(random.uniform(0.1, 0.25) * population_size)
	seeds = []
	for i in range(num_seeds):
		random_tree_index = random.choice([x for x in range(population_size) if x != current_tree_index])
		seed = [0 for x in range(dimensions)]
		for d in range(dimensions):
			space_min, space_max = search_space[d]
			if random.random() < search_tendency:
				seed[d] = trees[current_tree_index][d] + random.uniform(-1, 1) * (trees[best_tree_index][d] - trees[random_tree_index][d])
			else:
				seed[d] = trees[current_tree_index][d] + random.uniform(-1, 1) * (trees[current_tree_index][d] - trees[random_tree_index][d])
			if seed[d] > space_max:
				seed[d] = space_max
			if seed[d] < space_min:
				seed[d] = space_min
		seeds.append(seed)
	return seeds

def TSA(population_size, search_tendency, dimensions, f, search_space, train_set, test_set, filename):
	# maxFE = 10000 * dimensions			#maximum number of function evaluations
	# FE = 0 								#counter for function evaluations
	max_it = 10
	start_time = time.time()
	# print("Start time:", start_time)
	trees = initializeTrees(population_size, dimensions, search_space)
	# print(trees[0])
	string = "-" * 190 + "\nInitial values:" + "\n" + header1 + "\n" + header2 + "\n" + header3
	print(string)
	file = open(filename, 'a')
	file.write(string + "\n")
	file.close()
	# print("-" * 190)
	# print("Initial values:")
	# print(header1)
	# print(header2)
	# print(header3)
	trees_values, trees_times, trees_val_history, best_tree_index = chooseBest(trees, f, train_set, test_set, num_epochs, filename, verbose = True)
	# print("\t", trees_values)
	print("-" * 190)
	print()
	file = open(filename, 'a')
	file.write("-" * 190 + "\n\n")
	file.close()
	for it in range(max_it):
		string = "Iteration " + str(it+1) + ":\n" + header1 + "\n" + header2 + "\n" + header3
		print(string)
		file = open(filename, 'a')
		file.write(string + "\n")
		file.close()
		# print("Iteration:", it + 1)
		# print(header1)
		# print(header2)
		# print(header3)
		for current_tree_index in range(population_size):
			seeds = initializeSeeds(trees, current_tree_index, best_tree_index, population_size, search_tendency, dimensions, search_space)
			seeds_values, seeds_times, seeds_val_history, best_seed_index = chooseBest(seeds, f, train_set, test_set, num_epochs, filename, verbose = False)
			# best_seed_value, best_seed_time = CNN(seeds[best_seed_index], 5)
			if seeds_values[best_seed_index] > trees_values[current_tree_index]:
				trees[current_tree_index] = seeds[best_seed_index]
				trees_values[current_tree_index] = seeds_values[best_seed_index]
				trees_times[current_tree_index] = seeds_times[best_seed_index]
				trees_val_history[current_tree_index] = seeds_val_history[best_seed_index]
			displayTree(current_tree_index + 1, trees[current_tree_index], trees_values[current_tree_index], trees_times[current_tree_index], trees_val_history[current_tree_index], filename)
			# print("Progress:", current_tree_index + 1, "/", population_size, end = "\r")
		print("Time elapsed:", time.time() - start_time, "seconds")
		best_tree_val = max(trees_values)
		best_tree_index = trees_values.index(best_tree_val)
		# displayTrees(trees, trees_values, trees_times)
		print("-" * 190)
		print()
		file = open(filename, 'a')
		file.write("Time elapsed: " + str(time.time() - start_time) + " seconds\n" + "-" * 190 + "\n\n")
		file.close()
	# string = "Train for 5 epochs:\n" + header1 + "\n" + header2 + "\n" + header3
	# print(string)
	# file = open(filename, 'a')
	# file.write(string + "\n")
	# file.close()
	# # print("Train for 5 epochs:")
	# # print(header1)
	# # print(header2)
	# # print(header3)
	# trees_values, trees_times, best_tree_index = chooseBest(trees, f, train_set, test_set, 5, filename, verbose = True)
	# # displayTrees(trees, trees_values, trees_times)
	# print("-" * 190)
	# print("Total time:", time.time() - start_time, "seconds")
	# file = open(filename, 'a')
	# file.write("-" * 190 + "\nTotal time: " + str(time.time() - start_time) + " seconds")
	# file.close()
	return trees, trees_values, best_tree_index

def CNN(hyperparameters, train_set, test_set, epochs):
	train_images, train_labels = train_set
	test_images, test_labels = test_set
	hyperparameters = [round(parameter) for parameter in hyperparameters]
	C1_num_filter, C2_num_filter, C1_kernel_size, C2_kernel_size, C1_activation, C2_activation, FC1_activation, FC2_activation, FC1_neuron, FC2_neuron, batch_size, optimizer = hyperparameters
	C1_activation = activations[C1_activation]
	C2_activation = activations[C2_activation]
	FC1_activation = activations[FC1_activation]
	FC2_activation = activations[FC2_activation]
	C1_kernel_size = kernel_size[C1_kernel_size]
	C2_kernel_size = kernel_size[C2_kernel_size]
	optimizer = optimizers[optimizer]
	# with tf.device('/gpu:0'):
	start = time.time()
	model = models.Sequential()
	model.add(layers.Conv2D(C1_num_filter, C1_kernel_size, activation=C1_activation, input_shape=x_train.shape[1:]))
	model.add(layers.AveragePooling2D(2))
	# model.add(layers.Activation('sigmoid'))
	model.add(layers.Conv2D(C2_num_filter, C2_kernel_size, activation=C2_activation))
	model.add(layers.AveragePooling2D(2))
	# model.add(layers.Activation('sigmoid'))
	# model.add(layers.Conv2D(120, 5, activation='tanh'))
	model.add(layers.Flatten())
	model.add(layers.Dense(FC1_neuron, activation=FC1_activation))
	model.add(layers.Dense(FC2_neuron, activation=FC2_activation))
	model.add(layers.Dense(10, activation='softmax'))
	# model.summary()
	model.compile(optimizer=optimizer, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

	history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels), verbose=0)
	# loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
	end = time.time()
	return (history.history["val_accuracy"][-1], end-start, history.history["val_accuracy"])

def displayTrees(trees, values, times):
	print(f'{"".center(8)}|{"Results".center(25)}|{"C1 Layer".center(37)}|{"C3 Layer".center(37)}|{"F5 Layer".center(27)}|{"F6 Layer".center(27)}|{"General".center(22)}|')
	print(f'{"Tree".center(8)}|{"Accuracy".center(12)}|{"Running".center(12)}|{"Number of".center(13)}{"Kernel".center(10)}{"Activation".center(14)}|{"Number of".center(13)}{"Kernel".center(10)}{"Activation".center(14)}|{"Number of".center(13)}{"Activation".center(14)}|{"Number of".center(13)}{"Activation".center(14)}|{"Batch".center(9)}{"Optimizer".center(13)}|')
	print(f'{"".center(8)}|{"".center(12)}|{"time".center(12)}|{"filters".center(13)}{"size".center(10)}{"function".center(14)}|{"filters".center(13)}{"size".center(10)}{"function".center(14)}|{"neurons".center(13)}{"function".center(14)}|{"neurons".center(13)}{"function".center(14)}|{"size".center(9)}{"".center(13)}|')
	for i in range(len(trees)):
		parameters = trees[i]
		parameters = [round(parameter) for parameter in parameters]
		C1_num_filter, C2_num_filter, C1_kernel_size, C2_kernel_size, C1_activation, C2_activation, FC1_activation, FC2_activation, FC1_neuron, FC2_neuron, batch_size, optimizer = parameters
		C1_activation = activations[C1_activation]
		C2_activation = activations[C2_activation]
		FC1_activation = activations[FC1_activation]
		FC2_activation = activations[FC2_activation]
		C1_kernel_size = kernel_size[C1_kernel_size]
		C2_kernel_size = kernel_size[C2_kernel_size]
		optimizer = optimizers[optimizer]
		print(f'{str(i+1).center(8)}|{str(sf.round(values[i], sigfigs=8)).center(12)}|{str(sf.round(times[i], sigfigs=8)).center(12)}|{str(C1_num_filter).center(13)}{str(C1_kernel_size).center(10)}{C1_activation.center(14)}|{str(C2_num_filter).center(13)}{str(C2_kernel_size).center(10)}{C2_activation.center(14)}|{str(FC1_neuron).center(13)}{FC1_activation.center(14)}|{str(FC2_neuron).center(13)}{FC2_activation.center(14)}|{str(batch_size).center(9)}{optimizer.center(13)}|')

def displayTree(index, tree, value, time, val_history, filename):
	parameters = [round(parameter) for parameter in tree]
	C1_num_filter, C2_num_filter, C1_kernel_size, C2_kernel_size, C1_activation, C2_activation, FC1_activation, FC2_activation, FC1_neuron, FC2_neuron, batch_size, optimizer = parameters
	C1_activation = activations[C1_activation]
	C2_activation = activations[C2_activation]
	FC1_activation = activations[FC1_activation]
	FC2_activation = activations[FC2_activation]
	C1_kernel_size = kernel_size[C1_kernel_size]
	C2_kernel_size = kernel_size[C2_kernel_size]
	optimizer = optimizers[optimizer]
	string = f'{str(index).center(8)}|{str(sf.round(value, sigfigs=8)).center(12)}|'
	for val in val_history:
		string += f'{str(sf.round(val, sigfigs=8)).center(12)}|'
	string += f'{str(sf.round(time, sigfigs=8)).center(12)}|{str(C1_num_filter).center(13)}{str(C1_kernel_size).center(10)}{C1_activation.center(14)}|{str(C2_num_filter).center(13)}{str(C2_kernel_size).center(10)}{C2_activation.center(14)}|{str(FC1_neuron).center(13)}{FC1_activation.center(14)}|{str(FC2_neuron).center(13)}{FC2_activation.center(14)}|{str(batch_size).center(9)}{optimizer.center(13)}|'
	print(string)
	file = open(filename, 'a')
	file.write(string + '\n')
	file.close()

#search space for trees
search_space = [#range				description
				(3.5, 100.5),		#number of filters in C1
				(3.5, 100.5),		#number of filters in C2
				(-0.5, 2.5),			#C1 kernel size
				(-0.5, 2.5),			#C2 kernel size
				(-0.5, 2.5),			#C1 activation function
				(-0.5, 2.5),			#C2 activation function
				(-0.5, 2.5),			#FC1 activation function
				(-0.5, 2.5),			#FC2 activation function
				(3.5, 200),		#FC1 number of neurons
				(3.5, 200),		#FC2 number of neurons
				(9.5, 100.5),		#batch size
				(0, 1)]			#optimizer
#parameter options
activations = ["sigmoid", "relu", "tanh"]
optimizers = ["adam", "SGD"]
kernel_size = [3, 5, 7]

#dataset initialization
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
# x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
# x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255

x_train = x_train/255
x_test = x_test/255

# x_train = tf.expand_dims(x_train, axis=3, name=None)
# x_test = tf.expand_dims(x_test, axis=3, name=None)

#start algorithm
# print("MNIST dataset")
print("CIFAR10 dataset")
for trial in range(1):
    ST = 0.1
    population_size = 10
    filename = f'ST = {ST}, N = {population_size} ({trial + 1}).txt'
    file = open(filename, "a")
    # file.write("MNIST dataset\n")
    file.write("CIFAR10 dataset\n")
    file.close()
    TSA(population_size, ST, 12, CNN, search_space, (x_train, y_train), (x_test, y_test), filename)
