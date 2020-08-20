from ..chap_7.cnn_basic_model import *


class CnnRegModel(CnnBasicModel):
    def __init__(self, name, dataset, hconfigs, show_maps=False,
                 l2_decay=0, l1_decay=0):
        self.l2_decay = l2_decay
        self.l1_decay = l1_decay
        super(CnnRegModel, self).__init__(name, dataset, hconfigs, show_maps)

    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001,
                 report=0, show_cnt=3, show_params=False):
        super(CnnRegModel, self).exec_all(epoch_count, batch_size,
                                          learning_rate, report, show_cnt)
        if show_params:
            self.show_param_dist()

    def show_param_dist(self):
        params = self.collect_params()
        mu = np.mean(params)
        sigma = np.sqrt(np.var(params))
        plt.hist(params, 100, density=True, facecolor='g', alpha=0.75)
        plt.axis([-0.2, 0.2, 0, 20.0])
        plt.text(0.08, 15.0, 'mu={:5.3f}'.format(mu))
        plt.text(0.08, 13.0, 'sigma={:5.3f}'.format(sigma))
        plt.grid(True)
        plt.show()

        total_count = len(params)
        near_zero_count = len(list(x for x in params if -1e-5 <= x <= 1e-5))
        print('Near 0 parameters = {:4.1f}%({}/{})'.
              format(near_zero_count / total_count * 100, near_zero_count, total_count))

    def collect_params(self):
        params = list(self.pm_output['w'].flatten())
        for pm in self.pm_hiddens:
            if 'w' in pm:
                params += list(pm['w'].flatten())
            if 'k' in pm:
                params += list(pm['k'].flatten())
        return params

    def forward_extra_cost(self, y):
        extra, aux_extra = super(CnnRegModel, self).forward_extra_cost(y)
        if self.l2_decay > 0 or self.l1_decay > 0:
            params = self.collect_params()
            if self.l2_decay > 0:
                extra += np.sum(np.square(params)) / 2
            if self.l1_decay > 0:
                extra += np.sum(np.abs(params))
        return extra, aux_extra

    def update_param(self, pm, key, delta):
        if self.use_adam:
            delta = self.eval_adam_delta(pm, key, delta)

        if key in ['w', 'k']:
            if self.l2_decay > 0:
                delta += self.l2_decay * pm[key]
            if self.l1_decay > 0:
                delta += self.l1_decay * np.sign(pm[key])

        pm[key] -= self.learning_rate * delta

    @staticmethod
    def alloc_dropout_layer(input_shape, hconfig):
        keep_prob = get_conf_param(hconfig, 'keep_prob', 1.0)
        assert keep_prob > 0 and keep_prob <= 1
        return {'keep_prob': keep_prob}, input_shape

    def forward_dropout_layer(self, x, hconfig, pm):
        if self.is_training:
            dmask = np.random.binomial(1, pm['keep_prob'], x.shape)
            dropped = x * dmask / pm['keep_prob']
            return dropped, dmask
        else:
            return x, None

    @staticmethod
    def backprop_dropout_layer(G_y, hconfig, pm, aux):
        dmask = aux
        G_hidden = G_y * dmask / pm['keep_prob']
        return G_hidden

    @staticmethod
    def alloc_noise_layer(input_shape, hconfig):
        noise_type = get_conf_param(hconfig, 'type', 'normal')
        mean = get_conf_param(hconfig, 'mean', 0)
        std = get_conf_param(hconfig, 'std', 1.0)
        ratio = get_conf_param(hconfig, 'ratio', 1.0)

        assert noise_type == 'normal'

        return {'mean': mean, 'std': std, 'ratio': ratio}, input_shape

    def forward_noise_layer(self, x, hconfig, pm):
        if self.is_training and np.random.rand() < pm['ratio']:
            noise = np.random.normal(pm['mean'], pm['std'], x.shape)
            return x + noise, None
        else:
            return x, None

    @staticmethod
    def backprop_noise_layer(G_y, hconfig, pm, aux):
        return G_y

    @staticmethod
    def alloc_batch_normal_layer(input_shape, hconfig):
        pm = {}
        rescale = get_conf_param(hconfig, 'rescale', True)
        pm['epsilon'] = get_conf_param(hconfig, 'epsilon', 1e-10)
        pm['exp_ratio'] = get_conf_param(hconfig, 'exp_ratio', 0.001)

        bn_dim = input_shape[-1]
        pm['mavg'] = np.zeros(bn_dim)
        pm['mvar'] = np.ones(bn_dim)
        if rescale:
            pm['scale'] = np.ones(bn_dim)
            pm['shift'] = np.zeros(bn_dim)
        return pm, input_shape

    def forward_batch_normal_layer(self, x, hconfig, pm):
        if self.is_training:
            x_flat = x.reshape([-1, x.shape[-1]])
            avg = np.mean(x_flat, axis=0)
            var = np.var(x_flat, axis=0)
            pm['mavg'] += pm['exp_ratio'] * (avg - pm['mavg'])
            pm['mvar'] += pm['exp_ratio'] * (var - pm['mvar'])
        else:
            avg = pm['mavg']
            var = pm['mvar']
        std = np.sqrt(var + pm['epsilon'])
        y = norm_x = (x - avg) / std
        if 'scale' in pm:
            y = pm['scale'] * norm_x + pm['shift']
        return y, [norm_x, std]

    def backprop_batch_normal_layer(self, G_y, hconfig, pm, aux):
        norm_x, std = aux
        if 'scale' in pm:
            if len(G_y.shape) == 2:
                axis = 0
            else:
                axis = (0, 1, 2)
            G_scale = np.sum(G_y * norm_x, axis=axis)
            G_shift = np.sum(G_y, axis=axis)
            G_y = G_y * pm['scale']
            pm['scale'] -= self.learning_rate * G_scale
            pm['shift'] -= self.learning_rate * G_shift
        G_input = G_y / std
        return G_input
