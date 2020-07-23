import os
import numpy as np
import tensorflow as tf
from imageio import imread
from .attribution import *
from .utils import *
from .networks import *
from .visualize import visualize_attribution_maps


class PerturbedIterators:
    @staticmethod
    def get_pixelcnn(image, K, N, p_X_ij, S=1):
        """

        :param np.ndarray image:
        :param int K: size of patch
        :param int N: number of MC samples to approximate expectation
        :param p_X_ij: (i, j) -> [K, K, 3, 256] function
        :param int S: stride

        :return:
        """
        H, W, C = image.shape

        if K == 16:
            B = min(8, N)

        else:
            B = min(64, N)

        image_stacked = np.tile(np.expand_dims(image, axis=0), (B, 1, 1, 1))  # [N, H, W, C]

        def gen():
            for i in range(0, H - K + 1, S):
                for j in range(0, W - K + 1, S):
                    p_X = p_X_ij(i, j)
                    mask = np.zeros((H, W), dtype=np.bool)
                    mask[i: i + K, j: j + K] = True

                    for _ in range(int(N // B)):
                        perturbed = image_stacked.copy()
                        for pi, pj in ranges(K, K):
                            for c in range(C):
                                perturbed[:, i + pi, j + pj, c] = np.random.choice(256, size=B, p=p_X[pi, pj, c])

                        yield perturbed, mask

        output_types = (tf.int32, tf.bool)
        output_shapes = ((B, H, W, C), (H, W))
        dataset = tf.data.Dataset.from_generator(gen, output_types, output_shapes)
        dataset = dataset.prefetch(8)

        return dataset.make_initializable_iterator()  # perturbed, mask


class AttributionConfig:
    def __init__(self, image_path, K, S, N):
        self.image_path = image_path
        self.image = imread(image_path)

        self.K = K
        self.S = S
        self.N = N


def get_prediction(image, ckpt_path, Model):
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        with task('Build prediction_fn'):
            x = tf.placeholder(tf.uint8, [None, 224, 224, 3], name='image')
            classifier = Model(x)
            classifier.load(sess, ckpt_path)
            probs_op2 = classifier.probs

            def predict_fn(images):
                if isinstance(images, list):
                    images = np.array(images)

                return sess.run(probs_op2, feed_dict={x: images})[..., :1000]

            p_Y_x = predict_fn([image])[0]  # [Y], p_Y_x of original image
            classes = np.argsort(p_Y_x)[-3:][::-1]

    tf.reset_default_graph()
    return p_Y_x, classes


def _calculate(config):
    image = config.image
    H, W, C = image.shape

    ckpt_path = 'ckpts/vgg19/vgg19.ckpt'
    p_Y_x, classes = get_prediction(image, ckpt_path, VGG19)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        K = config.K
        N = config.N
        S = config.S

        with task('1. Get p_X'):
            X = tf.placeholder(tf.float32, [None, 3 * K, 3 * K, 3])
            model = PatchSampler(X, None, k=K, c=3)
            model.load(sess)
            image_padded = pad_images(np.expand_dims(image, axis=0), K)

            def p_X_ij(i, j):
                patch = image_padded[:, i: i + 3 * K, j: j + 3 * K]
                p_x_ij = sess.run(model.probs, feed_dict={X: patch})
                return p_x_ij[0]

        with task('2. Build graph'):
            
            iterator = PerturbedIterators.get_pixelcnn(image, K, N, p_X_ij=p_X_ij, S=S)

            X_perturb, mask_perturb = iterator.get_next()

            classifier = VGG19(X_perturb)
            run_op = {
                'probs': classifier.probs,
                'perturbated_image': X_perturb,
                'mask': mask_perturb
            }

        def update(mask_o_, p_Y_Xpert_o_):
            if p_Y_Xpert_o_.shape[0] == 0:
                return

            for cls in classes:
                attr.compute('PMI', ITAttr.PMI_MC, p_Y_x, p_Y_Xpert_o_, mask_o_, cls)

            attr.compute('IG', ITAttr.IG_MC, p_Y_x, p_Y_Xpert_o_, mask_o_)

        with task('4. Computing p_Y_Xperts'):
            classifier.load(sess, ckpt_path)
            sess.run(iterator.initializer)
            attr = AttrCalculator(H, W)

            mask_o = 0
            p_Y_Xpert_o = np.zeros((0, 1000), dtype=np.float32)
            for i_batch, result in runner(sess, run_op):
                p_Y_Xpert = result['probs'][..., :1000]
                mask = result['mask']

                if (mask != mask_o).any():
                    update(mask_o, p_Y_Xpert_o)
                    p_Y_Xpert_o = np.zeros((0, 1000), dtype=np.float32)
                    mask_o = mask

                p_Y_Xpert_o = np.concatenate([p_Y_Xpert_o, p_Y_Xpert])
            else:
                update(mask_o, p_Y_Xpert_o)

    with task('5. Get attribution'):
        PMImaps = [attr.get_result('PMI', cls) for cls in classes]
        IGmap = attr.get_result('IG')

    result = {
        'image': image,
        'p_Y_x': p_Y_x[classes],
        'PMImaps': PMImaps,
        'IGmap': IGmap,
        'classes': classes,
        'class_names': [get_imagenet_classname(cls) for cls in classes]
    }
    return result


def save_map(config):
    image_path = config.image_path
    name = os.path.basename(image_path).split('.')[0]
    K = config.K
    N = config.N
    S = config.S
    fpath = f'data/results/{name}_K{K}_N{N}_S{S}.pkl'
    if os.path.exists(fpath):
        print('Output file already exists. skip')
        return

    with task('Calculate'):
        result = _calculate(config)
        save_binary(result, fpath)
        print(result['p_Y_x'], result['class_names'])
        imagepath = f'data/results/{name}_K{K}_N{N}_S{S}.png'
        visualize_attribution_maps(result, imagepath)
        print('Results saved at "data/results/"')
