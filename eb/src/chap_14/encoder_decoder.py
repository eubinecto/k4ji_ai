from ..chap_12.rnn_ext_model import *


class EncoderDecoder(RnnExtModel):
    pass

    def init_parameters(self, hconfigs):
        econf = hconfigs['encoder']
        dconf = hconfigs['decoder']

        in_shape = self.dataset.input_shape

        pme, code_shape = self.build_subnet(econf, in_shape)
        pmd, hidden_shape = self.build_subnet(dconf, code_shape)

        self.econfigs, self.dconfigs = econf, dconf
        self.pm_encoder, self.pm_decoder = pme, pmd

    def build_subnet(self, hconfigs, prev_shape):
        pms = []

        for hconfig in hconfigs:
            pm, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            pms.append(pm)

        return pms, prev_shape

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode
        self.dataset.set_train_mode(train_mode)

    def step(self, epoch_count=10, batch_size=10, learning_rate=0.001,
                   report=0, show_cnt=3, train_mode='both'):
        self.set_train_mode(train_mode)
        self.train(epoch_count, batch_size, learning_rate, report)

    def exec_1_step(self, epoch_count=10, batch_size=10,
                          learning_rate=0.001, report=0, show_cnt=3):
        self.step(epoch_count, batch_size, learning_rate, report, show_cnt, 'both')
        self.test()
        if show_cnt > 0: self.visualize(show_cnt)

    def exec_2_steps(self, epoch_count=10, batch_size=10,
                           learning_rate=0.001, report=0, show_cnt=3):
        self.step(epoch_count, batch_size, learning_rate, report, 0, 'encoder')
        self.step(epoch_count, batch_size, learning_rate, report, show_cnt, 'decoder')
        self.set_train_mode('both')
        self.test()
        if show_cnt > 0: self.visualize(show_cnt)

    def exec_3_steps(self, epoch_count=10, batch_size=10,
                           learning_rate=0.001, report=0, show_cnt=3):
        self.step(epoch_count, batch_size, learning_rate, report, 0, 'encoder')
        self.step(epoch_count, batch_size, learning_rate, report, 0, 'decoder')
        self.step(epoch_count, batch_size, learning_rate, report, show_cnt, 'both')
        self.test()
        if show_cnt > 0: self.visualize(show_cnt)

    def forward_neuralnet(self, x):
        hidden = x

        aux_encoder, aux_decoder = [], []

        if self.train_mode in ['both', 'encoder']:
            for n, hconfig in enumerate(self.econfigs):
                hidden, aux = self.forward_layer(hidden, hconfig, self.pm_encoder[n])
                aux_encoder.append(aux)

        if self.train_mode in ['both', 'decoder']:
            for n, hconfig in enumerate(self.dconfigs):
                hidden, aux = self.forward_layer(hidden, hconfig, self.pm_decoder[n])
                aux_decoder.append(aux)

        output = hidden

        return output, [aux_encoder, aux_decoder]

    def backprop_neuralnet(self, G_output, aux):
        aux_encoder, aux_decoder = aux

        G_hidden = G_output

        if self.train_mode in ['both', 'decoder']:
            for n in reversed(range(len(self.dconfigs))):
                hconfig, pm = self.dconfigs[n], self.pm_decoder[n]
                aux = aux_decoder[n]
                G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

        if self.train_mode in ['both', 'encoder']:
            for n in reversed(range(len(self.econfigs))):
                hconfig, pm = self.econfigs[n], self.pm_encoder[n]
                aux = aux_encoder[n]
                G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

        return G_hidden

    def visualize(self, num):
        print('Model {} Visualization'.format(self.name))
        self.set_train_mode('both')
        deX, deY = self.dataset.get_visualize_data(num)
        self.set_train_mode('encoder')
        code, _ = self.forward_neuralnet(deX)
        self.set_train_mode('decoder')
        output, _ = self.forward_neuralnet(code)
        self.dataset.visualize(deX, code, output, deY)