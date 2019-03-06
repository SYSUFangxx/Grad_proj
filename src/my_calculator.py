import numpy as np


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


def calc_weight(w_o, w_b, r, X, upper=0.02, c=0.006, l=100):
    # X: nstocks*nfactors
    # w_o: 1*nstocks
    # w_b: 1*nstocks
    # r: 1*nstocks

    y = np.dot(w_o - w_b, X)

    n = len(w_o)
    w = np.ones(n) / n
    v = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)
    q = np.zeros(n)
    z = np.ones(n) / n
    h = np.zeros(n)

    mu = 1e-4
    eps = 1e-8
    maxmu = 1e10
    rho = 1.02

    T = 0

    while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(u - v) > eps or np.linalg.norm(
                w - z) > eps) and T < 10000:

        # update v
        threshold = c / (2 * mu)
        pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)

        for i in range(0, n):
            v[i] = min(0.0, pj_vec[i] + threshold) + max(0.0, pj_vec[i] - threshold)

        part_r = r - q + mu * v - l * np.dot(y, X.T)
        part_l = l * np.dot(X, X.T) + mu * np.identity(n)
        u = np.dot(np.linalg.inv(part_l), part_r)
        w = simplex_projection((v + w_o + z) / 2 + (h - p) / (2 * mu))
        for i in range(0, n):
            z[i] = max(0.0, min(w[i] - h[i] / mu, upper))

        p = p + mu * (w - v - w_o)
        q = q + mu * (u - v)
        h = h + mu * (z - w)
        mu = min(mu * rho, maxmu)
        T = T + 1
        if T % 200 == 0 or (np.linalg.norm(w - w_o - v) <= eps and np.linalg.norm(u - v) <= eps and np.linalg.norm(
                    w - z) <= eps):
            total = 0
            for kk in range(0, n):
                total = total + abs(w[kk] - w_o[kk])
            p3 = np.linalg.norm(np.dot(w - w_b, X))
            p2 = np.dot(w, r.T)
            loss = c * total - p2 + 0.5 * l * p3 * p3
            print('iter {} loss {} cost {} return {} risk {}'.format(T, loss, total, p2, p3 * p3))

    return w


def calc_weight_with_exprosure(w_o, w_b, r, X, upper=0.02, c=0.006, exprosure=0.05):
    # X: nstocks*nfactors
    # w_o: 1*nstocks
    # w_b: 1*nstocks
    # r: 1*nstocks

    y = np.dot(w_o - w_b, X)

    n = len(w_o)
    m = X.shape[1]

    w = np.ones(n) / n
    v = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)
    q = np.zeros(n)
    z = np.ones(n) / n
    h = np.zeros(n)

    g = np.zeros(m)
    s = np.zeros(m)
    t_low = np.ones(m) * (-exprosure)
    t_upper = np.ones(m) * (exprosure)

    mu = 1e-4
    eps = 1e-8
    maxmu = 1e10
    rho = 1.02

    T = 0

    while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(np.dot(u, X) + y - s) > eps or np.linalg.norm(
                u - v) > eps or np.linalg.norm(w - z) > eps) and T < 10000:

        # update v
        threshold = c / (2 * mu)
        pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)

        for i in range(0, n):
            v[i] = min(0.0, pj_vec[i] + threshold) + max(0.0, pj_vec[i] - threshold)

        part1 = r - q + mu * v - mu * np.dot(y, X.T) + mu * np.dot(s, X.T) - np.dot(g, X.T)
        part2 = mu * np.dot(X, X.T) + mu * np.identity(n)
        u = np.dot(np.linalg.inv(part2), part1)
        # u = np.dot(part1, np.linalg.inv(part2))

        w = simplex_projection((v + w_o + z) / 2 - (h + p) / (2 * mu))

        for i in range(0, n):
            z[i] = max(0.0, min(w[i] + h[i] / mu, upper))

        tmp_vec = np.dot(u, X) + y + g / mu
        for i in range(0, m):
            s[i] = max(t_low[i], min(tmp_vec[i], t_upper[i]))

        p = p + mu * (w - v - w_o)
        q = q + mu * (u - v)
        h = h + mu * (w - z)
        g = g + mu * (np.dot(u, X) + y - s)
        mu = min(mu * rho, maxmu)
        T = T + 1
        if T % 200 == 0 or (np.linalg.norm(np.dot(u, X) + y - s) <= eps and np.linalg.norm(
                        w - w_o - v) <= eps and np.linalg.norm(u - v) <= eps and np.linalg.norm(w - z) <= eps):
            total = 0
            for kk in range(0, n):
                total = total + abs(w[kk] - w_o[kk])
            p3 = np.linalg.norm(np.dot(w - w_b, X))
            p2 = np.dot(w, r.T)
            loss = c * total - p2
            print('iter {} loss {} cost {} return {} risk_expro {}'.format(T, loss, total, p2, p3 * p3))

    return w
