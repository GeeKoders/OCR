import cv2
import config
import utils
import siamese_network
from keras.layers import Input
from keras.models import Model, load_model
from keras.layers import Lambda
from keras.optimizers import RMSprop

input_shape, tr_pairs, tr_y, te_pairs, te_y = utils.load_data()

# network definition
base_network = siamese_network.create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(siamese_network.euclidean_distance,output_shape=siamese_network.eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=siamese_network.contrastive_loss, optimizer=rms, metrics=[siamese_network.accuracy])
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=config.BATCH_SIZE,
          epochs=config.EPOCHS,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = siamese_network.compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = siamese_network.compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

model.save(config.MODEL_PATH)
utils.plot_training(history, config.PLOT_PATH)