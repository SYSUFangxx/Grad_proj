

def simplex_projection(v):
    v_sorted = sorted(v, reverse=True)
    # print(v_sorted[0:3])
    n = len(v)
    max_ = 0
    for j in range(1, n):
        sum_ = 1
        for k in range(0, j):
            sum_ -= (v_sorted[k] - v_sorted[j])
        if sum_ > 0:
            max_ = j
    max_ = max_ + 1

    sigma = 0
    for i in range(0, max_):
        sigma = sigma + v_sorted[i]
    sigma = (sigma - 1) / max_

    w = np.zeros(len(v))
    for i in range(0, n):
        w[i] = max(0.0, v[i] - sigma)
    return w
