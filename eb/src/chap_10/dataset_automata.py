from ..chap_5.dataset import *

MIN_LENGTH = 10
MAX_LENGTH = 40

ALPHA = [chr(n) for n in range(ord('a'), ord('z') + 1)]
DIGIT = [chr(n) for n in range(ord('0'), ord('9') + 1)]

EOS = ['$']
ADDOP = ['+', '-']
MULTOP = ['*', '/']
LPAREN = ['(']
RPAREN = [')']

SYMBOLS = EOS + ADDOP + MULTOP + LPAREN + RPAREN
ALPHANUM = ALPHA + DIGIT
ALPHABET = SYMBOLS + ALPHANUM

S = 0  # sent
E = 1  # exp
T = 2  # term
F = 3  # factor
V = 4  # variable
N = 5  # number
V2 = 6  # var_tail

RULES = {
    S: [[E]],
    E: [[T], [E, ADDOP, T]],
    T: [[F], [T, MULTOP, F]],
    F: [[V], [N], [LPAREN, E, RPAREN]],
    V: [[ALPHA], [ALPHA, V2]],
    V2: [[ALPHANUM], [ALPHANUM, V2]],
    N: [[DIGIT], [DIGIT, N]]
}

E_NEXT = EOS + RPAREN + ADDOP
T_NEXT = E_NEXT + MULTOP
F_NEXT = T_NEXT
V_NEXT = F_NEXT
N_NEXT = F_NEXT

action_table = {
    0: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    1: [[ADDOP, 9], [EOS, 0]],
    2: [[MULTOP, 10], [E_NEXT, -1, E]],
    3: [[T_NEXT, -1, T]],
    4: [[F_NEXT, -1, F]],
    5: [[F_NEXT, -1, F]],
    6: [[ALPHANUM, 6], [V_NEXT, -1, V]],
    7: [[DIGIT, 7], [N_NEXT, -1, N]],
    8: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    9: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    10: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    11: [[V_NEXT, -2, V]],
    12: [[N_NEXT, -2, N]],
    13: [[RPAREN, 16], [ADDOP, 9]],
    14: [[MULTOP, 10], [T_NEXT, -3, T]],
    15: [[F_NEXT, -3, F]],
    16: [[F_NEXT, -3, F]],
}

goto_table = {
    0: {E: 1, T: 2, F: 3, V: 4, N: 5},
    6: {V: 11},
    7: {N: 12},
    8: {E: 13, T: 2, F: 3, V: 4, N: 5},
    9: {T: 14, F: 3, V: 4, N: 5},
    10: {F: 15, V: 4, N: 5},
}


def automata_generate_sent():
    while True:
        try:
            sent = automata_gen_node(S, 0)
            if len(sent) >= MAX_LENGTH:
                continue
            if len(sent) <= MIN_LENGTH:
                continue
            return sent
        except Exception:
            continue


def automata_gen_node(node, depth):
    if depth > 30:
        raise Exception
    if node not in RULES:
        assert 0
    rules = RULES[node]
    nth = np.random.randint(len(rules))
    sent = ''
    for term in rules[nth]:
        if isinstance(term, list):
            pos = np.random.randint(len(term))
            sent += term[pos]
        else:
            sent += automata_gen_node(term, depth + 1)
    return sent


def automata_is_correct_sent(sent):
    sent = sent + '$'
    states, pos, nextch = [0], 0, sent[0]

    while True:
        actions = action_table[states[-1]]
        found = False
        for pair in actions:
            if nextch not in pair[0]:
                continue
            found = True
            if pair[1] == 0:  # accept
                return True
            elif pair[1] > 0:  # shift
                states.append(pair[1])
                pos += 1
                nextch = sent[pos]
                break
            else:  # reduce
                states = states[:pair[1]]
                goto = goto_table[states[-1]]
                goto_state = goto[pair[2]]
                states.append(goto_state)
                break
        if not found:  # error
            return False


def automata_generate_data(count):
    xs = np.zeros([count, MAX_LENGTH, len(ALPHABET)])
    ys = np.zeros([count, 1])

    for n in range(count):
        is_correct = n % 2

        if is_correct:
            sent = automata_generate_sent()
        else:
            while True:
                sent = automata_generate_sent()
                touch = np.random.randint(1, len(sent) // 5)
                for k in range(touch):
                    sent_pos = np.random.randint(len(sent))
                    char_pos = np.random.randint(len(ALPHABET) - 1)
                    sent = sent[:sent_pos] + ALPHABET[char_pos] + \
                           sent[sent_pos + 1:]
                if not automata_is_correct_sent(sent):
                    break

        ords = [ALPHABET.index(ch) for ch in sent]
        xs[n, 0, 0] = len(sent)
        xs[n, 1:len(sent) + 1, :] = np.eye(len(ALPHABET))[ords]
        ys[n, 0] = is_correct

    return xs, ys


class AutomataDataset(Dataset):
    def __init__(self):
        super(AutomataDataset, self).__init__('automata', 'binary')
        self.input_shape = [MAX_LENGTH + 1, len(ALPHABET)]
        self.output_shape = [1]

    @property
    def train_count(self):
        return 10000

    def get_train_data(self, batch_size, nth):
        return automata_generate_data(batch_size)

    def get_validate_data(self, count):
        return automata_generate_data(count)

    def get_visualize_data(self, count):
        return self.get_validate_data(count)

    def get_test_data(self):
        return automata_generate_data(1000)

    @staticmethod
    def visualize(xs, est, ans):
        for n in range(len(xs)):
            length = int(xs[n, 0, 0])
            sent = np.argmax(xs[n, 1:length + 1], axis=1)
            text = "".join([ALPHABET[letter] for letter in sent])

            answer, guess, result = '잘못된 패턴', '탈락추정', 'X'

            if ans[n][0] > 0.5:
                answer = '올바른 패턴'
            if est[n][0] > 0.5:
                guess = '합격추정'
            if ans[n][0] > 0.5 and est[n][0] > 0.5:
                result = 'O'
            if ans[n][0] < 0.5 and est[n][0] < 0.5:
                result = 'O'

            print('{}: {} => {}({:4.2f}) : {}'.
                  format(text, answer, guess, est[n][0], result))
