
from .mathutil import *


class Dataset(object):
    def __init__(self, name, mode):

        # init member variables
        self.name = name
        self.mode = mode
        self.tr_xs = None
        self.tr_ys = None
        self.te_xs = None
        self.te_ys = None
        self.va_xs = None
        self.va_ys = None
        self.indices = None
        self.va_indices = None
        self.input_shape = None
        self.output_shape = None

    def __str__(self):
        return '{}({}, {}+{}+{})'.format(self.name,
                                         self.mode,
                                         len(self.tr_xs),
                                         len(self.te_xs),
                                         len(self.va_xs))

    @property
    def train_count(self):
        return len(self.tr_xs)

    def get_train_data(self, batch_size, nth):
        from_idx = nth * batch_size
        to_idx = (nth + 1) * batch_size

        tr_X = self.tr_xs[self.indices[from_idx:to_idx]]
        tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]

        return tr_X, tr_Y

    def shuffle_train_data(self, size):
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)

    def get_test_data(self):
        return self.te_xs, self.te_ys

    def get_validate_data(self, count):
        self.va_indices = np.arange(len(self.va_xs))
        np.random.shuffle(self.va_indices)

        va_X = self.va_xs[self.va_indices[0:count]]
        va_Y = self.va_ys[self.va_indices[0:count]]

        return va_X, va_Y

    def shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
        data_count = len(xs)

        tr_cnt = int(data_count * tr_ratio / 10) * 10
        va_cnt = int(data_count * va_ratio)
        te_cnt = data_count - (tr_cnt + va_cnt)

        tr_from, tr_to = 0, tr_cnt
        va_from, va_to = tr_cnt, tr_cnt + va_cnt
        te_from, te_to = tr_cnt + va_cnt, data_count

        indices = np.arange(data_count)
        np.random.shuffle(indices)

        self.tr_xs = xs[indices[tr_from:tr_to]]
        self.tr_ys = ys[indices[tr_from:tr_to]]
        self.va_xs = xs[indices[va_from:va_to]]
        self.va_ys = ys[indices[va_from:va_to]]
        self.te_xs = xs[indices[te_from:te_to]]
        self.te_ys = ys[indices[te_from:te_to]]

        self.input_shape = xs[0].shape
        self.output_shape = ys[0].shape

        return indices[tr_from:tr_to], indices[va_from:va_to], indices[te_from:te_to]

    def forward_postproc(self, output, y, mode=None):
        if mode is None:
            mode = self.mode

        if mode == 'regression':
            diff = output - y
            square = np.square(diff)
            loss = np.mean(square)
            aux = diff
        elif mode == 'binary':
            entropy = sigmoid_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [y, output]
        elif mode == 'select':
            entropy = softmax_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [output, y, entropy]
        else:
            raise ValueError()

        return loss, aux

    def backprop_postproc(self, G_loss, aux, mode=None):
        if mode is None:
            mode = self.mode

        if mode == 'regression':
            diff = aux
            shape = diff.shape

            g_loss_square = np.ones(shape) / np.prod(shape)
            g_square_diff = 2 * diff
            g_diff_output = 1

            G_square = g_loss_square * G_loss
            G_diff = g_square_diff * G_square
            G_output = g_diff_output * G_diff
        elif mode == 'binary':
            y, output = aux
            shape = output.shape

            g_loss_entropy = np.ones(shape) / np.prod(shape)
            g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy
        elif mode == 'select':
            output, y, entropy = aux

            g_loss_entropy = 1.0 / np.prod(entropy.shape)
            g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy
        else:
            raise ValueError()

        return G_output

    def eval_accuracy(self, x, y, output, mode=None):
        if mode is None:
            mode = self.mode

        if mode == 'regression':
            mse = np.mean(np.square(output - y))
            accuracy = 1 - np.sqrt(mse) / np.mean(y)
        elif mode == 'binary':
            estimate = np.greater(output, 0)
            answer = np.equal(y, 1.0)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)
        elif mode == 'select':
            estimate = np.argmax(output, axis=1)
            answer = np.argmax(y, axis=1)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)
        else:
            raise ValueError()
        return accuracy

    def get_estimate(self, output, mode=None):
        if mode is None:
            mode = self.mode

        if mode == 'regression':
            estimate = output
        elif mode == 'binary':
            estimate = sigmoid(output)
        elif mode == 'select':
            estimate = softmax(output)
        else:
            raise ValueError()

        return estimate

    @staticmethod
    def train_prt_result(epoch, costs, accs, acc, time1, time2):
        print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}/{:5.3f} ({}/{} secs)'
              .format(epoch, np.mean(costs), np.mean(accs), acc, time1, time2))

    @staticmethod
    def test_prt_result(name, acc, given_time):
        print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'
              .format(name, acc, given_time))
