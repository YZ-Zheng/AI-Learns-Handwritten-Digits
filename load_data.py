import gzip, _pickle, numpy as np

img_rows, img_cols = 42, 28


def get_data(path_to_data_dir):
	"""
	load the datasets
	"""
	f = gzip.open(path_to_data_dir + 'train_multi_digit' + '_mini' + '.pkl.gz', 'rb')
	X_train = _pickle.load(f, encoding='latin1')
	f.close()
	X_train =  np.reshape(X_train, (len(X_train), 1, img_rows, img_cols))
	f = gzip.open(path_to_data_dir + 'test_multi_digit' + '_mini' +'.pkl.gz', 'rb')
	X_test = _pickle.load(f, encoding='latin1')
	f.close()
	X_test =  np.reshape(X_test, (len(X_test),1, img_rows, img_cols))
	f = gzip.open(path_to_data_dir + 'train_labels' + '_mini' +'.txt.gz', 'rb')
	y_train = np.loadtxt(f)
	f.close()
	f = gzip.open(path_to_data_dir +'test_labels' + '_mini' + '.txt.gz', 'rb')
	y_test = np.loadtxt(f)
	f.close()
	return X_train, y_train, X_test, y_test
