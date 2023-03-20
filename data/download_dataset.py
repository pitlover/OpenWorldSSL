from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training data:")
print("Number of examples: ", X_train.shape[0])
print("Number of channels:", X_train.shape[3])
print("Image size:", X_train.shape[1], X_train.shape[2])
print
print("Test data:")
print("Number of examples:", X_test.shape[0])
print("Number of channels:", X_test.shape[3])
print("Image size:", X_test.shape[1], X_test.shape[2])
