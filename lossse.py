import tensorflow as tf

backend = tf.keras.backend


def def_iou( y_true, y_pred, label):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = backend.cast(y_true, tf.uint8)
    # compute cross entropy
    y_pred = backend.reshape(y_pred, shape=(-1, y_pred.shape[-1]))
    # y_pred = tf.keras.layers.Softmax(axis=-1)(y_pred)

    # y_pred = tf.argmax(y_pred,1)
    y_true = tf.reshape(y_true, shape=(-1,))

    # y_true_one_hot = tf.one_hot(y_true, 8)

    # y_true = backend.cast(backend.equal(backend.argmax(y_true_one_hot), label), backend.floatx())
    y_true = backend.cast(backend.equal(y_true, label), backend.floatx())

    y_pred = backend.cast(backend.equal(backend.argmax(y_pred), label), backend.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = backend.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = backend.sum(y_true) + backend.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return backend.switch(backend.equal(union, 0), 1.0, intersection / union)


def def_mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = backend.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    mean_iou = backend.variable(0)
    
    class_iou_dict = {}
    # iterate over labels to calculate IoU for
    for label in range(0, num_labels):
        label_iou = def_iou(y_true, y_pred, label)
        if label in class_iou_dict.keys():
            class_iou_dict[label] = class_iou_dict[label] + label_iou
        else:
            class_iou_dict[label] = label_iou
        mean_iou = mean_iou + label_iou
    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels,class_iou_dict

def categorical_crossentropy_with_logits(y_true, y_pred):

    mean_iou_loss = 1 - def_mean_iou(y_true, y_pred)[0]
    y_true =  backend.cast(y_true, tf.uint8)
    # compute cross entropy
    y_pred = backend.reshape(y_pred, shape=(-1, y_pred.shape[-1]))
    # y_pred = tf.keras.layers.Softmax(axis=-1)(y_pred)

    # y_pred = tf.argmax(y_pred,1)
    y_true = tf.reshape(y_true, shape = (-1,))

    y_true_one_hot = tf.one_hot(y_true,8)

    # cross_entropy = backend.categorical_crossentropy(y_true_one_hot, y_pred, from_logits=True)
    # print(cross_entropy.shape)
    # print(cross_entropy.shape)
    cross_entropy = tf.keras.losses.CategoricalCrossentropy()(y_true_one_hot, y_pred)

    # compute loss
    # loss = backend.mean(cross_entropy)
    total_loss = cross_entropy + mean_iou_loss
    return total_loss


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)
        # compute ce loss
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)
        # compute weights
        weights = backend.sum(alpha * backend.pow(1 - y_pred, gamma) * y_true, axis=-1)
        return backend.mean(backend.sum(weights * cross_entropy, axis=[1, 2]))

    return loss


def miou_loss(weights=None, num_classes=2):
    if weights is not None:
        assert len(weights) == num_classes
        weights = tf.convert_to_tensor(weights)
    else:
        weights = tf.convert_to_tensor([1.] * num_classes)

    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)

        inter = y_pred * y_true
        inter = backend.sum(inter, axis=[1, 2])

        union = y_pred + y_true - (y_pred * y_true)
        union = backend.sum(union, axis=[1, 2])

        return -backend.mean((weights * inter) / (weights * union + 1e-8))

    return loss


def self_balanced_focal_loss(alpha=3, gamma=2.0):
    """
    Original by Yang Lu:
    This is an improvement of Focal Loss, which has solved the problem
    that the factor in Focal Loss failed in semantic segmentation.
    It can adaptively adjust the weights of different classes in semantic segmentation
    without introducing extra supervised information.
    :param alpha: The factor to balance different classes in semantic segmentation.
    :param gamma: The factor to balance different samples in semantic segmentation.
    :return:
    """

    def loss(y_true, y_pred):
        # cross entropy loss
        y_pred = backend.softmax(y_pred, -1)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred)

        # sample weights
        sample_weights = backend.max(backend.pow(1.0 - y_pred, gamma) * y_true, axis=-1)

        # class weights
        pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                                   axis=[1, 2], keepdims=True)
        class_weights = backend.max(backend.pow(backend.ones_like(y_true) * alpha, pixel_rate) * y_true, axis=-1)

        # final loss
        final_loss = class_weights * sample_weights * cross_entropy
        return backend.mean(backend.sum(final_loss, axis=[1, 2]))

    return loss
