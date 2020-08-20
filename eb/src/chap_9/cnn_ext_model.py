from ..chap_8.cnn_reg_model import *


def check_add_shapes(yshape, bshape):
    assert yshape[:-1] == bshape[:-1]
    assert yshape[-1] % bshape[-1] == 0


def tile_add_result(by, ychn, bchn):
    if ychn == bchn: return by
    times = ychn // bchn
    return np.tile(by, times)


def merge_add_grad(G_y, ychn, bchn):
    if ychn == bchn: return G_y
    times = ychn // bchn
    split_shape = G_y.shape[:-1] + tuple([times, bchn])
    return np.sum(G_y.reshape(split_shape), axis=-2)


def replace_arg(exp, args):
    if isinstance(exp, (list, tuple)):
        for n, term in enumerate(exp):
            if isinstance(term, str) and term[0] == '#':
                if term[1] == '#': exp[n] = term[1:]
                elif term in args: exp[n] = args[term]
            else:
                replace_arg(term, args)
    elif isinstance(exp, dict):
        for key in exp:
            if isinstance(exp[key], str) and exp[key][0] == '#':
                if exp[key][1] == '#': exp[key] = exp[key][1:]
                elif exp[key] in args: exp[key] = args[exp[key]]
            else:
                replace_arg(exp[key], args)

def get_shape_params(hconfig, conv_type):
    if conv_type:
        kh, kw = get_conf_param_2d(hconfig, 'ksize')
        sh, sw = get_conf_param_2d(hconfig, 'stride', [1,1])
    else:
        sh, sw = get_conf_param_2d(hconfig, 'stride', [1,1])
        kh, kw = get_conf_param_2d(hconfig, 'ksize', [sh, sw])
    padding = get_conf_param(hconfig, 'padding', 'SAME')

    return kh, kw, sh, sw, padding


def eval_stride_shape(hconfig, conv_type, xh, xw, ychn):
    kh, kw, sh, sw, padding = get_shape_params(hconfig, conv_type)
    if padding == 'VALID':
        xh = xh - kh + 1
        xw = xw - kw + 1
    yh = xh // sh
    yw = xw // sw
    return [yh, yw, ychn]


def stride_filter(hconfig, conv_type, y):
    _, xh, xw, _ = x_shape = y.shape
    nh, nw = xh, xw
    kh, kw, sh, sw, padding = get_shape_params(hconfig, conv_type)

    if padding == 'VALID':
        bh, bw = (kh - 1) // 2, (kw - 1) // 2
        nh, nw = xh - kh + 1, xw - kw + 1
        y = y[:, bh:bh + nh, bw:bw + nw:, :]

    if sh != 1 or sw != 1:
        bh, bw = (sh - 1) // 2, (sw - 1) // 2
        mh, mw = nh // sh, nw // sw
        y = y[:, bh:bh + mh * sh:sh, bw:bw + mw * sw:sw, :]

    return y, [x_shape, nh, nw]


def stride_filter_derv(hconfig, conv_type, G_y, aux):
    x_shape, nh, nw = aux
    mb_size, xh, xw, chn = x_shape
    kh, kw, sh, sw, padding = get_shape_params(hconfig, conv_type)

    if sh != 1 or sw != 1:
        bh, bw = (sh - 1) // 2, (sw - 1) // 2
        mh, mw = nh // sh, nw // sw
        G_y_tmp = np.zeros([mb_size, nh, nw, chn])
        G_y_tmp[:, bh:bh + mh * sh:sh, bw:bw + mw * sw:sw, :] = G_y
        G_y = G_y_tmp

    if padding == 'VALID':
        bh, bw = (kh - 1) // 2, (kw - 1) // 2
        nh, nw = xh - kh + 1, xw - kw + 1
        G_y_tmp = np.zeros([mb_size, xh, xw, chn])
        G_y_tmp[:, bh:bh + nh, bw:bw + nw:, :] = G_y
        G_y = G_y_tmp

    return G_y


class CnnExtModel(CnnRegModel):
    macros = {}

    def __init__(self, name, dataset, hconfigs, show_maps=False,
                 l2_decay=0, l1_decay=0, dump_structure=False):
        self.dump_structure = dump_structure
        self.layer_index = 0
        self.layer_depth = 0
        self.param_count = 0
        super(CnnExtModel, self).__init__(name, dataset, hconfigs, show_maps,
                                          l2_decay, l1_decay)
        if self.dump_structure:
            print('Total parameter count: {}'.format(self.param_count))

    def alloc_layer_param(self, input_shape, hconfig):
        layer_type = get_layer_type(hconfig)

        if layer_type in ['serial', 'parallel', 'loop', 'add', 'custom']:
            if self.dump_structure:
                dump_str = layer_type
                if layer_type == 'custom':
                    name = get_conf_param(hconfig, 'name')
                    dump_str += ' ' + name
                print('{:>{width}}{}'.format('', dump_str,
                                             width=self.layer_depth * 2))
            self.layer_depth += 1

        pm, output_shape = super(CnnExtModel, self). \
            alloc_layer_param(input_shape, hconfig)

        if layer_type in ['serial', 'parallel', 'loop', 'add', 'custom']:
            self.layer_depth -= 1
        elif self.dump_structure:
            self.layer_index += 1
            pm_str = ''
            if layer_type == 'full':
                ph, pw = pm['w'].shape
                pm_count = np.prod(pm['w'].shape) + pm['b'].shape[0]
                self.param_count += pm_count
                pm_str = ' pm:{}x{}+{}={}'.format(ph, pw, pm['b'].shape[0], pm_count)
            elif layer_type == 'conv':
                kh, kw, xchn, ychn = pm['k'].shape
                pm_count = np.prod(pm['k'].shape) + pm['b'].shape[0]
                self.param_count += pm_count
                pm_str = ' pm:{}x{}x{}x{}+{}={}'.format(kh, kw, xchn, ychn,
                                                        pm['b'].shape[0], pm_count)
            print('{:>{width}}{}: {}, {}=>{}{}'. \
                  format('', self.layer_index, layer_type, input_shape,
                         output_shape, pm_str, width=self.layer_depth * 2))

        return pm, output_shape

    def alloc_parallel_layer(self, input_shape, hconfig):
        pm_hiddens = []
        output_shape = None

        if not isinstance(hconfig[1], dict): hconfig.insert(1, {})

        for bconfig in hconfig[2:]:
            bpm, bshape = self.alloc_layer_param(input_shape, bconfig)
            pm_hiddens.append(bpm)
            if output_shape:
                assert output_shape[0:-1] == bshape[0:-1]
                output_shape[-1] += bshape[-1]
            else:
                output_shape = bshape

        return {'pms': pm_hiddens}, output_shape

    def forward_parallel_layer(self, x, hconfig, pm):
        bys, bauxes, bchns = [], [], []
        for n, bconfig in enumerate(hconfig[2:]):
            by, baux = self.forward_layer(x, bconfig, pm['pms'][n])
            bys.append(by)
            bauxes.append(baux)
            bchns.append(by.shape[-1])
        y = np.concatenate(bys, axis=-1)
        return y, [bauxes, bchns]

    def backprop_parallel_layer(self, G_y, hconfig, pm, aux):
        bauxes, bchns = aux
        bcn_from = 0
        G_x = 0
        for n, bconfig in enumerate(hconfig[2:]):
            bcn_to = bcn_from + bchns[n]
            G_y_slice = G_y[:, :, :, bcn_from:bcn_to]
            G_x += self.backprop_layer(G_y_slice, bconfig, pm['pms'][n], bauxes[n])
        return G_x

    def alloc_serial_layer(self, input_shape, hconfig):
        pm_hiddens = []
        prev_shape = input_shape

        if not isinstance(hconfig[1], dict): hconfig.insert(1, {})

        for sconfig in hconfig[2:]:
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, sconfig)
            pm_hiddens.append(pm_hidden)

        return {'pms': pm_hiddens}, prev_shape

    def forward_serial_layer(self, x, hconfig, pm):
        hidden = x
        auxes = []

        for n, sconfig in enumerate(hconfig[2:]):
            hidden, aux = self.forward_layer(hidden, sconfig, pm['pms'][n])
            auxes.append(aux)

        return hidden, auxes

    def backprop_serial_layer(self, G_y, hconfig, pm, aux):
        auxes = aux
        G_hidden = G_y

        for n in reversed(range(len(hconfig[2:]))):
            sconfig, spm, saux = hconfig[2:][n], pm['pms'][n], auxes[n]
            G_hidden = self.backprop_layer(G_hidden, sconfig, spm, saux)

        return G_hidden

    def alloc_add_layer(self, input_shape, hconfig):
        if not isinstance(hconfig[1], dict): hconfig.insert(1, {})

        bpm, output_shape = self.alloc_layer_param(input_shape, hconfig[2])
        pm_hiddens = [bpm]

        for bconfig in hconfig[3:]:
            bpm, bshape = self.alloc_layer_param(input_shape, bconfig)
            pm_hiddens.append(bpm)
            check_add_shapes(output_shape, bshape)

        if get_conf_param(hconfig, 'x', True):
            check_add_shapes(output_shape, input_shape)

        pm = {'pms': pm_hiddens}

        for act in get_conf_param(hconfig, 'actions', ''):
            if act == 'B':
                bn_config = ['batch_normal', {'rescale': True}]
                pm['bn'], _ = self.alloc_batch_normal_layer(output_shape,
                                                            bn_config)

        return pm, output_shape

    def forward_add_layer(self, x, hconfig, pm):
        y, baux = self.forward_layer(x, hconfig[2], pm['pms'][0])
        bauxes, bchns, aux_bn = [baux], [y.shape[-1]], []

        for n, bconfig in enumerate(hconfig[3:]):
            by, baux = self.forward_layer(x, bconfig, pm['pms'][n + 1])
            y += tile_add_result(by, y.shape[-1], by.shape[-1])
            bauxes.append(baux)
            bchns.append(by.shape[-1])

        if get_conf_param(hconfig, 'x', True):
            y += tile_add_result(x, y.shape[-1], x.shape[-1])

        for act in get_conf_param(hconfig, 'actions', ''):
            if act == 'A': y = self.activate(y, hconfig)
            if act == 'B':
                y, aux_bn = self.forward_batch_normal_layer(y, None, pm['bn'])

        return y, [y, bauxes, bchns, aux_bn, x.shape]

    def backprop_add_layer(self, G_y, hconfig, pm, aux):
        y, bauxes, bchns, aux_bn, x_shape = aux

        for act in reversed(get_conf_param(hconfig, 'actions', '')):
            if act == 'A': G_y = self.activate_derv(G_y, y, hconfig)
            if act == 'B':
                G_y = self.backprop_batch_normal_layer(G_y, None, pm['bn'],
                                                       aux_bn)

        G_x = np.zeros(x_shape)

        for n, bconfig in enumerate(hconfig[2:]):
            G_by = merge_add_grad(G_y, G_y.shape[-1], bchns[n])
            G_x += self.backprop_layer(G_by, bconfig, pm['pms'][n], bauxes[n])

        if get_conf_param(hconfig, 'x', True):
            G_x += merge_add_grad(G_y, G_y.shape[-1], x_shape[-1])

        return G_x

    def alloc_loop_layer(self, input_shape, hconfig):
        pm_hiddens = []
        prev_shape = input_shape

        if not isinstance(hconfig[1], dict): hconfig.insert(1, {})

        for n in range(get_conf_param(hconfig, 'repeat', 1)):
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig[2])
            pm_hiddens.append(pm_hidden)

        return {'pms': pm_hiddens}, prev_shape

    def forward_loop_layer(self, x, hconfig, pm):
        hidden = x
        aux_layers = []

        for n in range(get_conf_param(hconfig, 'repeat', 1)):
            hidden, aux = self.forward_layer(hidden, hconfig[2], pm['pms'][n])
            aux_layers.append(aux)

        return hidden, aux_layers

    def backprop_loop_layer(self, G_y, hconfig, pm, aux):
        G_hidden = G_y
        aux_layers = aux

        for n in reversed(range(get_conf_param(hconfig, 'repeat', 1))):
            pm_hidden, aux = pm['pms'][n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig[2], pm_hidden, aux)

        return G_hidden

    def alloc_custom_layer(self, input_shape, hconfig):
        name = get_conf_param(hconfig, 'name')
        args = get_conf_param(hconfig, 'args', {})
        macro = CnnExtModel.get_macro(name, args)

        pm_hidden, output_shape = self.alloc_layer_param(input_shape, macro)

        return {'pm': pm_hidden, 'macro': macro}, output_shape

    def forward_custom_layer(self, x, hconfig, pm):
        return self.forward_layer(x, pm['macro'], pm['pm'])

    def backprop_custom_layer(self, G_y, hconfig, pm, aux):
        return self.backprop_layer(G_y, pm['macro'], pm['pm'], aux)

    def set_macro(name, config):
        CnnExtModel.macros[name] = config

    def get_macro(name, args):
        restored = copy.deepcopy(CnnExtModel.macros[name])
        replace_arg(restored, args)

        return restored

    def alloc_conv_layer(self, input_shape, hconfig):
        pm, output_shape = super(CnnExtModel, self). \
            alloc_conv_layer(input_shape, hconfig)
        pm['actions'] = get_conf_param(hconfig, 'actions', 'LA')
        for act in pm['actions']:
            if act == 'L':
                input_shape = output_shape
            elif act == 'B':
                bn_config = ['batch_normal', {'rescale': False}]
                pm['bn'], _ = self.alloc_batch_normal_layer(input_shape, bn_config)

        xh, xw, xchn = input_shape
        ychn = get_conf_param(hconfig, 'chn')
        output_shape = eval_stride_shape(hconfig, True, xh, xw, ychn)
        return pm, output_shape

    def forward_conv_layer(self, x, hconfig, pm):
        y = x
        x_flat, k_flat, relu_y, aux_bn = None, None, None, None
        for act in pm['actions']:
            if act == 'L':
                mb_size, xh, xw, xchn = y.shape
                kh, kw, _, ychn = pm['k'].shape
                x_flat = get_ext_regions_for_conv(y, kh, kw)
                k_flat = pm['k'].reshape([kh * kw * xchn, ychn])
                conv_flat = np.matmul(x_flat, k_flat)
                y = conv_flat.reshape([mb_size, xh, xw, ychn]) + pm['b']
            elif act == 'A':
                y = self.activate(y, hconfig)
                relu_y = y
            elif act == 'B':
                y, aux_bn = self.forward_batch_normal_layer(y, None, pm['bn'])

        y, aux_stride = stride_filter(hconfig, True, y)

        if self.need_maps: self.maps.append(y)

        return y, [x_flat, k_flat, x, relu_y, aux_bn, aux_stride]

    def backprop_conv_layer(self, G_y, hconfig, pm, aux):
        x_flat, k_flat, x, relu_y, aux_bn, aux_stride = aux

        G_x = stride_filter_derv(hconfig, True, G_y, aux_stride)

        for act in reversed(pm['actions']):
            if act == 'L':
                kh, kw, xchn, ychn = pm['k'].shape
                mb_size, xh, xw, _ = G_x.shape

                G_conv_flat = G_x.reshape(mb_size * xh * xw, ychn)
                g_conv_k_flat = x_flat.transpose()
                g_conv_x_flat = k_flat.transpose()
                G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
                G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
                G_bias = np.sum(G_conv_flat, axis=0)
                G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
                G_x = undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

                self.update_param(pm, 'k', G_kernel)
                self.update_param(pm, 'b', G_bias)
            elif act == 'A':
                G_x = self.activate_derv(G_x, relu_y, hconfig)
            elif act == 'B':
                G_x = self.backprop_batch_normal_layer(G_x, None, pm['bn'], aux_bn)

        return G_x

    @staticmethod
    def alloc_max_layer(input_shape, hconfig):
        xh, xw, ychn = input_shape
        output_shape = eval_stride_shape(hconfig, False, xh, xw, ychn)
        return None, output_shape

    def forward_max_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride', [1, 1])
        kh, kw = get_conf_param_2d(hconfig, 'ksize', [sh, sw])
        padding = get_conf_param(hconfig, 'padding', 'SAME')

        if [sh, sw] == [kh, kw] and xh % sh == 0 and xw % sw == 0 and \
                padding == 'SAME':
            return super(CnnExtModel, self).forward_max_layer(x, hconfig, pm)

        x_flat = get_ext_regions(x, kh, kw, -np.inf)
        x_flat = x_flat.transpose([2, 5, 0, 1, 3, 4])
        x_flat = x_flat.reshape(mb_size * chn * xh * xw, kh * kw)

        max_idx = np.argmax(x_flat, axis=1)
        y = x_flat[np.arange(x_flat.shape[0]), max_idx]
        y = y.reshape([mb_size, chn, xh, xw])
        y = y.transpose([0, 2, 3, 1])

        y, aux_stride = stride_filter(hconfig, False, y)

        if self.need_maps: self.maps.append(y)

        return y, [x.shape, kh, kw, sh, sw, padding, max_idx, aux_stride]

    def backprop_max_layer(self, G_y, hconfig, pm, aux):
        if not isinstance(aux, list):
            return super(CnnExtModel, self).backprop_max_layer(G_y, hconfig,
                                                               pm, aux)

        x_shape, kh, kw, sh, sw, padding, max_idx, aux_stride = aux
        mb_size, xh, xw, chn = x_shape

        G_y = stride_filter_derv(hconfig, False, G_y, aux_stride)

        G_y = G_y.transpose([0, 3, 1, 2])
        G_y = G_y.flatten()

        G_x_flat = np.zeros([mb_size * chn * xh * xw, kh * kw])
        G_x_flat[np.arange(G_x_flat.shape[0]), max_idx] = G_y

        G_x_flat = G_x_flat.reshape(mb_size, chn, xh, xw, kh, kw)
        G_x_flat = G_x_flat.transpose([2, 3, 0, 4, 5, 1])
        G_x = undo_ext_regions(G_x_flat, kh, kw)

        return G_x

    def alloc_avg_layer(self, input_shape, hconfig):
        xh, xw, chn = input_shape
        sh, sw = get_conf_param_2d(hconfig, 'stride', [1, 1])
        kh, kw = get_conf_param_2d(hconfig, 'ksize', [sh, sw])
        padding = get_conf_param(hconfig, 'padding', 'SAME')

        if [sh, sw] == [kh, kw] and xh % sh == 0 and xw % sw == 0 \
                and padding == 'SAME':
            return super(CnnExtModel, self).alloc_avg_layer(input_shape, hconfig)

        one_mask = np.ones([1, xh, xw, chn])

        m_flat = get_ext_regions(one_mask, kh, kw, 0)
        m_flat = m_flat.transpose([2, 5, 0, 1, 3, 4])
        m_flat = m_flat.reshape(1 * chn * xh * xw, kh * kw)

        mask = np.sum(m_flat, axis=1)

        output_shape = eval_stride_shape(hconfig, False, xh, xw, chn)

        return {'mask': mask}, output_shape

    def forward_avg_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride', [1, 1])
        kh, kw = get_conf_param_2d(hconfig, 'ksize', [sh, sw])
        padding = get_conf_param(hconfig, 'padding', 'SAME')

        if [sh, sw] == [kh, kw] and xh % sh == 0 and xw % sw == 0 \
                and padding == 'SAME':
            return super(CnnExtModel, self).forward_avg_layer(x, hconfig, pm)

        x_flat = get_ext_regions(x, kh, kw, 0)
        x_flat = x_flat.transpose([2, 5, 0, 1, 3, 4])
        x_flat = x_flat.reshape(mb_size * chn * xh * xw, kh * kw)

        hap = np.sum(x_flat, axis=1)

        y = np.reshape(hap, [mb_size, -1]) / pm['mask']
        y = y.reshape([mb_size, chn, xh, xw])
        y = y.transpose([0, 2, 3, 1])

        y, aux_stride = stride_filter(hconfig, False, y)

        if self.need_maps: self.maps.append(y)

        return y, [x.shape, kh, kw, sh, sw, padding, aux_stride]

    def backprop_avg_layer(self, G_y, hconfig, pm, aux):
        if not isinstance(aux, list):
            return super(CnnExtModel, self).backprop_avg_layer(G_y, hconfig, pm, aux)

        x_shape, kh, kw, sh, sw, padding, aux_stride = aux
        mb_size, xh, xw, chn = x_shape

        G_y = stride_filter_derv(hconfig, False, G_y, aux_stride)

        G_y = G_y.transpose([0, 3, 1, 2])
        G_y = G_y.flatten()

        G_hap = np.reshape(G_y, [mb_size, -1]) / pm['mask']
        G_x_flat = np.tile(G_hap, (kh * kw, 1))

        G_x_flat = G_x_flat.reshape(mb_size, chn, xh, xw, kh, kw)
        G_x_flat = G_x_flat.transpose([2, 3, 0, 4, 5, 1])
        G_x = undo_ext_regions(G_x_flat, kh, kw)

        return G_x
