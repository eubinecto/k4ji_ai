from ..chap_11.rnn_lstm_model import *
import numpy as np


class RnnExtModel(RnnLstmModel):
    def alloc_seqwrap_layer(self, input_shape, hconfig):
        pms = []
        prev_shape = input_shape[1:]

        if not isinstance(hconfig[1], dict):
            hconfig.insert(1, {})

        for hconfig in hconfig[2:]:
            pm, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            pms.append(pm)

        if isinstance(prev_shape, int):
            prev_shape = [prev_shape]
        output_shape = [input_shape[0]] + list(prev_shape)

        return pms, output_shape

    def forward_seqwrap_layer(self, x, hconfig, pm):
        mb_size, timesteps1, rest = x.shape[0], x.shape[1], x.shape[2:]

        lengths = x.reshape([x.shape[0], -1])[:, 0]
        xbody = x[:, 1:].reshape([-1] + list(rest))

        pms = pm
        hidden = xbody
        aux_layers = []

        for n, hconfig in enumerate(hconfig[2:]):
            hidden, aux_h = self.forward_layer(hidden, hconfig, pms[n])
            aux_layers.append(aux_h)

        y_shape = tuple([mb_size, timesteps1]) + hidden.shape[1:]

        y = np.zeros([mb_size, timesteps1, np.prod(y_shape[2:])])
        y[:, 0, 0] = lengths
        y[:, 1:, :] = hidden.reshape([mb_size, timesteps1 - 1, -1])
        y = y.reshape(y_shape)

        return y, [lengths, x.shape, hidden.shape, aux_layers]

    def backprop_seqwrap_layer(self, G_y, hconfig, pm, aux):
        mb_size, timesteps1, rest = G_y.shape
        lengths, x_shape, h_shape, aux_layers = aux

        G_y_body = G_y.reshape([mb_size, timesteps1, -1])[:, 1:, :]
        G_hidden = G_y_body.reshape(h_shape)

        for n in reversed(range(len(hconfig[2:]))):
            config_h, pm_h, aux_h = hconfig[n + 2], pm[n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, config_h, pm_h, aux_h)

        G_x = np.zeros([mb_size, timesteps1, np.prod(x_shape[2:])])
        G_x[:, 0, 0] = lengths
        G_x[:, 1:, :] = G_hidden.reshape([mb_size, timesteps1 - 1, -1])
        G_input = G_x.reshape(x_shape)

        return G_input

    def init_parameters(self, hconfigs):
        self.hconfigs = hconfigs
        self.pm_hiddens = []

        prev_shape = self.dataset.input_shape

        for hconfig in hconfigs:
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            self.pm_hiddens.append(pm_hidden)

        output_cnt = int(np.prod(self.dataset.output_shape))
        self.seqout = False

        if len(hconfigs) > 0 and get_layer_type(hconfigs[-1]) in ['rnn', 'lstm']:
            if get_conf_param(hconfigs[-1], 'outseq', True):
                self.seqout = True
                prev_shape = prev_shape[1:]
                output_cnt = int(np.prod(self.dataset.output_shape[1:]))

        self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)

    def forward_neuralnet(self, x):
        hidden = x
        aux_layers = []

        for n, hconfig in enumerate(self.hconfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])
            aux_layers.append(aux)

        if self.seqout:
            hshape = hidden.shape
            mb_size, timesteps = hshape[0], hshape[1] - 1

            hidden_temp = hidden.reshape([mb_size, timesteps + 1, -1])
            lengths = hidden_temp[:, 0, 0]
            hidden_flat = hidden_temp[:, 1:, :].reshape(mb_size * timesteps, -1)

            output_size = self.pm_output['w'].shape[1]

            out_flat, aux_flat = self.forward_layer(hidden_flat, None,
                                                    self.pm_output)

            output = np.zeros([mb_size, timesteps + 1, output_size])
            output[:, 0, 0] = lengths
            output[:, 1:, :] = out_flat.reshape([mb_size, timesteps, output_size])
            aux_out = [aux_flat, hshape]
        else:
            output, aux_out = self.forward_layer(hidden, None, self.pm_output)

        return output, [aux_out, aux_layers]

    def backprop_neuralnet(self, G_output, aux):
        aux_out, aux_layers = aux

        if self.seqout:
            aux_flat, hshape = aux_out
            mb_size, timesteps = hshape[0], hshape[1] - 1
            output_size = self.pm_output['w'].shape[1]
            G_out_flat = G_output[:, 1:, :].reshape([mb_size * timesteps, output_size])
            G_hidden_flat = self.backprop_layer(G_out_flat, None,
                                                self.pm_output, aux_flat)
            G_hidden = np.zeros(hshape)
            G_hidden[:, 0, 0] = G_output[:, 0, 0]
            G_hidden[:, 1:, :] = G_hidden_flat.reshape([mb_size, timesteps, -1])
        else:
            G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)

        for n in reversed(range(len(self.hconfigs))):
            hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

        return G_hidden
