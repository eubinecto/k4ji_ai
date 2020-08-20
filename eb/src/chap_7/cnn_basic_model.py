from ..chap_6.adam_model import *


def get_layer_type(hconfig):
    if not isinstance(hconfig, list):
        return 'full'
    return hconfig[0]


def get_conf_param(hconfig, key, defval=None):
    if not isinstance(hconfig, list):
        return defval
    if len(hconfig) <= 1:
        return defval
    if key not in hconfig[1]:
        return defval
    return hconfig[1][key]


def get_conf_param_2d(hconfig, key, defval=None):
    if len(hconfig) <= 1:
        return defval
    if key not in hconfig[1]:
        return defval
    val = hconfig[1][key]
    if isinstance(val, list):
        return val
    return [val, val]


def get_ext_regions_for_conv(x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    regs = get_ext_regions(x, kh, kw, 0)
    regs = regs.transpose([2, 0, 1, 3, 4, 5])

    return regs.reshape([mb_size * xh * xw, kh * kw * xchn])


def get_ext_regions(x, kh, kw, fill):
    mb_size, xh, xw, xchn = x.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    x_ext = np.zeros((mb_size, eh, ew, xchn), dtype='float32') + fill
    x_ext[:, bh:bh + xh, bw:bw + xw, :] = x

    regs = np.zeros((xh, xw, mb_size * kh * kw * xchn), dtype='float32')

    for r in range(xh):
        for c in range(xw):
            regs[r, c, :] = x_ext[:, r:r + kh, c:c + kw, :].flatten()

    return regs.reshape([xh, xw, mb_size, kh, kw, xchn])


def undo_ext_regions_for_conv(regs, x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    regs = regs.reshape([mb_size, xh, xw, kh, kw, xchn])
    regs = regs.transpose([1, 2, 0, 3, 4, 5])

    return undo_ext_regions(regs, kh, kw)


def undo_ext_regions(regs, kh, kw):
    xh, xw, mb_size, kh, kw, xchn = regs.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    gx_ext = np.zeros([mb_size, eh, ew, xchn], dtype='float32')

    for r in range(xh):
        for c in range(xw):
            gx_ext[:, r:r + kh, c:c + kw, :] += regs[r, c]

    return gx_ext[:, bh:bh + xh, bw:bw + xw, :]


class CnnBasicModel(AdamModel):

    def __init__(self, name, dataset, hconfigs, show_maps=False):
        if isinstance(hconfigs, list) and \
                not isinstance(hconfigs[0], (list, int)):
            hconfigs = [hconfigs]
        self.maps = None
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []
        super(CnnBasicModel, self).__init__(name, dataset, hconfigs)
        self.use_adam = True

    def alloc_layer_param(self, input_shape, hconfig):
        layer_type = get_layer_type(hconfig)

        m_name = 'alloc_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        pm, output_shape = method(input_shape, hconfig)

        return pm, output_shape

    def forward_layer(self, x, hconfig, pm):
        layer_type = get_layer_type(hconfig)

        m_name = 'forward_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        y, aux = method(x, hconfig, pm)

        return y, aux

    def backprop_layer(self, G_y, hconfig, pm, aux):
        layer_type = get_layer_type(hconfig)

        m_name = 'backprop_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        G_input = method(G_y, hconfig, pm, aux)

        return G_input

    def alloc_full_layer(self, input_shape, hconfig):
        input_cnt = np.prod(input_shape)
        output_cnt = get_conf_param(hconfig, 'width', hconfig)

        weight = np.random.normal(0, self.rand_std, [input_cnt, output_cnt])
        bias = np.zeros([output_cnt])

        return {'w': weight, 'b': bias}, [output_cnt]

    def alloc_conv_layer(self, input_shape, hconfig):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        kh, kw = get_conf_param_2d(hconfig, 'ksize')
        ychn = get_conf_param(hconfig, 'chn')

        kernel = np.random.normal(0, self.rand_std, [kh, kw, xchn, ychn])
        bias = np.zeros([ychn])

        if self.show_maps:
            self.kernels.append(kernel)

        return {'k': kernel, 'b': bias}, [xh, xw, ychn]

    @staticmethod
    def alloc_pool_layer(input_shape, hconfig):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')

        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]

    def forward_full_layer(self, x, hconfig, pm):
        if pm is None:
            return x, None

        x_org_shape = x.shape

        if len(x.shape) != 2:
            mb_size = x.shape[0]
            x = x.reshape([mb_size, -1])

        affine = np.matmul(x, pm['w']) + pm['b']
        y = self.activate(affine, hconfig)

        return y, [x, y, x_org_shape]

    def backprop_full_layer(self, G_y, hconfig, pm, aux):
        if pm is None:
            return G_y

        x, y, x_org_shape = aux

        G_affine = self.activate_derv(G_y, y, hconfig)

        g_affine_weight = x.transpose()
        g_affine_input = pm['w'].transpose()

        G_weight = np.matmul(g_affine_weight, G_affine)
        G_bias = np.sum(G_affine, axis=0)
        G_input = np.matmul(G_affine, g_affine_input)

        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)

        return G_input.reshape(x_org_shape)

    @staticmethod
    def activate(affine, hconfig):
        if hconfig is None:
            return affine

        func = get_conf_param(hconfig, 'actfunc', 'relu')

        if func == 'none':
            return affine
        elif func == 'relu':
            return relu(affine)
        elif func == 'sigmoid':
            return sigmoid(affine)
        elif func == 'tanh':
            return tanh(affine)
        else:
            assert 0

    @staticmethod
    def activate_derv(G_y, y, hconfig):
        if hconfig is None:
            return G_y

        func = get_conf_param(hconfig, 'actfunc', 'relu')

        if func == 'none':
            return G_y
        elif func == 'relu':
            return relu_derv(y) * G_y
        elif func == 'sigmoid':
            return sigmoid_derv(y) * G_y
        elif func == 'tanh':
            return tanh_derv(y) * G_y
        else:
            assert 0

    def forward_conv_layer_adhoc(self, x, hconfig, pm):
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        conv = np.zeros((mb_size, xh, xw, ychn))

        for n in range(mb_size):
            for r in range(xh):
                for c in range(xw):
                    for ym in range(ychn):
                        for i in range(kh):
                            for j in range(kw):
                                rx = r + i - (kh - 1) // 2
                                cx = c + j - (kw - 1) // 2
                                if rx < 0 or rx >= xh:
                                    continue
                                if cx < 0 or cx >= xw:
                                    continue
                                for xm in range(xchn):
                                    kval = pm['k'][i][j][xm][ym]
                                    ival = x[n][rx][cx][xm]
                                    conv[n][r][c][ym] += kval * ival

        y = self.activate(conv + pm['b'], hconfig)

        return y, [x, y]

    def forward_conv_layer_better(self, x, hconfig, pm):
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        conv = np.zeros((mb_size, xh, xw, ychn))

        bh, bw = (kh - 1) // 2, (kw - 1) // 2
        eh, ew = xh + kh - 1, xw + kw - 1

        x_ext = np.zeros((mb_size, eh, ew, xchn))
        x_ext[:, bh:bh + xh, bw:bw + xw, :] = x

        k_flat = pm['k'].transpose([3, 0, 1, 2]).reshape([ychn, -1])

        for n in range(mb_size):
            for r in range(xh):
                for c in range(xw):
                    for ym in range(ychn):
                        xe_flat = x_ext[n, r:r + kh, c:c + kw, :].flatten()
                        conv[n, r, c, ym] = (xe_flat * k_flat[ym]).sum()

        y = self.activate(conv + pm['b'], hconfig)

        return y, [x, y]

    def forward_conv_layer(self, x, hconfig, pm):
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        x_flat = get_ext_regions_for_conv(x, kh, kw)
        k_flat = pm['k'].reshape([kh * kw * xchn, ychn])
        conv_flat = np.matmul(x_flat, k_flat)
        conv = conv_flat.reshape([mb_size, xh, xw, ychn])

        y = self.activate(conv + pm['b'], hconfig)

        if self.need_maps:
            self.maps.append(y)

        return y, [x_flat, k_flat, x, y]

    def backprop_conv_layer(self, G_y, hconfig, pm, aux):
        x_flat, k_flat, x, y = aux

        kh, kw, xchn, ychn = pm['k'].shape
        mb_size, xh, xw, _ = G_y.shape

        G_conv = self.activate_derv(G_y, y, hconfig)

        G_conv_flat = G_conv.reshape(mb_size * xh * xw, ychn)

        g_conv_k_flat = x_flat.transpose()
        g_conv_x_flat = k_flat.transpose()

        G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
        G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
        G_bias = np.sum(G_conv_flat, axis=0)

        G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
        G_input = undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

        self.update_param(pm, 'k', G_kernel)
        self.update_param(pm, 'b', G_bias)

        return G_input

    def forward_avg_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        y_flat = np.average(x3, 1)
        y = y_flat.reshape([mb_size, yh, yw, chn])

        if self.need_maps:
            self.maps.append(y)

        return y, None

    @staticmethod
    def backprop_avg_layer(G_y, hconfig, pm, aux):
        mb_size, yh, yw, chn = G_y.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten() / (sh * sw)

        gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
        for i in range(sh * sw):
            gx1[:, i] = gy_flat
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input

    def forward_max_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        idxs = np.argmax(x3, axis=1)
        y_flat = x3[np.arange(mb_size * yh * yw * chn), idxs]
        y = y_flat.reshape([mb_size, yh, yw, chn])

        if self.need_maps:
            self.maps.append(y)

        return y, idxs

    @staticmethod
    def backprop_max_layer(G_y, hconfig, pm, aux):
        idxs = aux

        mb_size, yh, yw, chn = G_y.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten()

        gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
        gx1[np.arange(mb_size * yh * yw * chn), idxs] = gy_flat[:]
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input

    def visualize(self, num):
        print('Model {} Visualization'.format(self.name))

        self.need_maps = self.show_maps
        self.maps = []

        deX, deY = self.dataset.get_visualize_data(num)
        est = self.get_estimate(deX)

        if self.show_maps:
            for kernel in self.kernels:
                kh, kw, xchn, ychn = kernel.shape
                grids = kernel.reshape([kh, kw, -1]).transpose(2, 0, 1)
                draw_images_horz(grids[0:5, :, :])

            for pmap in self.maps:
                draw_images_horz(pmap[:, :, :, 0])

        self.dataset.visualize(deX, est, deY)

        self.need_maps = False
        self.maps = None
