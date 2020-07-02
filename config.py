import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def read_image(filename):
  image_string = tf.read_file(filename)
  image = tf.image.decode_jpeg(image_string, channels=3)
  # convert to float values in [0, 1]
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def scale_image_value(image):
  # scale values between -1 and +1
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def resize_image(image):
  resized_image = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])
  return resized_image


def resize_and_random_crop_image(image):
  minval = FLAGS.image_size # 224
  maxval = 280
  new_height = tf.random_uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.int32)
  new_width = new_height
  resized_image = tf.image.resize_images(image, [new_height, new_width])
  crop_image = tf.random_crop(resized_image, [FLAGS.image_size, FLAGS.image_size, 3])
  return crop_image


def train_image_parse_function(filename, *argv):
  """ preprocess image for training. """
  image = read_image(filename)
  image = tf.image.random_flip_left_right(image)

  if FLAGS.augmentation:
    print('data augmentation')
    resized_image = resize_and_random_crop_image(image)
  else:
    resized_image = resize_image(image)
  resized_image = scale_image_value(resized_image)

  if len(argv) == 1:
    return resized_image, argv[0]
  elif len(argv) == 2:
    return resized_image, argv[0], argv[1]
  else:
    return resized_image


def train_pair_image_parse_function(filename1, filename2, *argv):
  """ preprocess image for training. """
  resized_image1 = train_image_parse_function(filename1)
  resized_image2 = train_image_parse_function(filename2)

  if len(argv) == 1:
    return resized_image1, resized_image2, argv[0]
  elif len(argv) == 2:
    return resized_image1, resized_image2, argv[0], argv[1]
  elif len(argv) == 3:
    return resized_image1, resized_image2, argv[0], argv[1], argv[2]
  else:
    return resized_image1, resized_image2


def eval_image_parse_function(filename, text):
  """ preprocess image for evaluation. """
  image = read_image(filename)
  resized_image = resize_image(image)
  resized_image = scale_image_value(resized_image)
  return resized_image, text


def optimistic_restore_vars(model_checkpoint_path):
  reader = tf.train.NewCheckpointReader(model_checkpoint_path)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
        restore_vars.append(curr_var)
  return restore_vars
