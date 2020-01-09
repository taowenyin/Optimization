import numpy as np
import common as com

if __name__ == '__main__':
    q = [
        [2, 0],
        [0, 1]
    ]

    x = [
        [1],
        [2]
    ]

    g = com.rank_one_gradient(q, x)
    print(g)

    d = com.rank_one_direction(com.create_unit_matrix(2), g)
    print(d)

    a = com.rank_one_step(g, d, q)
    print(a)

    x = com.rank_one_x(x, a, d)
    print(x)