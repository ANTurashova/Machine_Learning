"""
Турашова Анна, вариант 11.

Дан шифртекст. Используя алфавит, приведенный в примере (используется кривая E_751(-1,1) и
генерирующая точка G = (–1, 1)), и зная секретный ключ n_b, найти открытый текст.
"""

n_b = 27
a = -1
cipher_text = [[[745, 210], [185, 105]], [[188, 93], [681, 385]], [[377, 456], [576, 465]], [[440, 539], [138, 298]],
[[745, 210], [520, 2]], [[188, 93], [681, 385]], [[286, 136], [282, 410]], [[72, 254], [200, 721]],
[[72, 254], [643, 94]], [[745, 210], [476, 315]], [[440, 539], [724, 229]]]
b = 1
p = 751
x, y = 0, 0


def find_lambda(x1, y1, x2, y2):
    if x1 % p == x2 % p and y1 % p == y2 % p:
        return (((3 * (x1 * x1)) + a) * inverse_to_element((2 * y1) % p, p)) % p
    elif x1 % p == x2 % p:
        return 0
    else:
        return ((y2 - y1) * inverse_to_element((x2 - x1) % p, p)) % p


def euclidean_algorithm(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, x, y = euclidean_algorithm(b % a, a)
        return g, y - (b // a) * x, x


def inverse_to_element(a, n):
    g, x, _ = euclidean_algorithm(a, n)
    if g == 1:
        return x % n


for cipher in cipher_text:
    for n in range(1, n_b):
        if n == 1:
            f_lambda = find_lambda(cipher[0][0], cipher[0][1], cipher[0][0], cipher[0][1])
            x = (((f_lambda * f_lambda) - cipher[0][0]) - cipher[0][0]) % p
            y = ((f_lambda * (cipher[0][0] - x)) - cipher[0][1]) % p
        else:
            f_lambda = find_lambda(cipher[0][0], cipher[0][1], x, y)
            x = (((f_lambda * f_lambda) - cipher[0][0]) - x) % p
            y = ((f_lambda * (cipher[0][0] - x)) - cipher[0][1]) % p

    y = (-1 * y) % p

    f_lambda = find_lambda(cipher[1][0], cipher[1][1], x, y)
    x = (((f_lambda * f_lambda) - cipher[1][0]) - x) % p
    y = ((f_lambda * (cipher[1][0] - x)) - cipher[1][1]) % p

    print(f'({x}, {y})')

print('Ответ: летательный')
