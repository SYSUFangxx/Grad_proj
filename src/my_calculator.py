import numpy as np


def simplex_projection(v):
    """
    这里对应John Duchi 2008年的论文中的figure1算法，用于求解单纯形上的欧式投影问题
    注意：
        z=1
        rho与j跟原论文的下标可能有些小差异，因为程序中数组是从0开始计数的
    :param v: 约束目标中的向量v
    :return: 算法求解的向量w
    """
    z = 1.
    v_sorted = sorted(v, reverse=True)
    n = len(v)
    rho = 0
    for j in range(1, n):
        sum_mur = 0
        for r in range(j):
            sum_mur += v_sorted[r]
        if v_sorted[j - 1] - (sum_mur - z) / j > 0:
            rho = j
        else:
            # if j == 1:
            #     print(list(v))
            #     print(v_sorted)
            #     print('*' * 50)
            break

    sum_mui = 0
    for i in range(rho):
        sum_mui += v_sorted[i]
    theta = (sum_mui - z) / rho

    w = np.zeros(len(v))
    for i in range(n):
        w[i] = max(0.0, v[i] - theta)
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

    mu = 1e-12
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
        # print(T)

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


class ReferPO:
    def __init__(self):
        pass

    @staticmethod
    def olu(inputs):
        """
        paper: Online Lazy Updates for Portfolio Selection with Transaction Costs
        :param inputs: 算法的输入
                x_t:    相对价格向量，size = 1 * stocks
                w_o:    原来的组合权重，size = 1 * stocks
        :return: w(t+1)
        """
        # get inputs
        x_t = inputs['x_t']
        w_o = inputs['w_o']

        # iteration constant
        eps = 1e-4
        max_iter = 10000

        # algorithm constant
        beta = 0.01
        eta = 1
        gamma = 0.002
        alpha = eta * gamma

        # initialize
        n = len(w_o)
        w = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)

        k = 0
        while (np.linalg.norm(w - w_o - z) > eps or np.linalg.norm(np.sum(w) - 1) > eps) and k < max_iter:
            # iterate p(k+1), z(k+1), u(k+1)
            part1 = -1 * eta / ((beta + 1) * np.sum(w_o * x_t)) * x_t
            part2 = w_o
            part3 = beta / (beta + 1) * z
            part4 = -1 * beta / (beta + 1) * u
            w = simplex_projection(part1 + part2 + part3 + part4)

            threshold = alpha / beta
            proj_vec = w - w_o + u
            for i in range(n):
                z = min(0.0, proj_vec[i] + threshold) + max(0.0, proj_vec[i] - threshold)

            u = u + (w - w_o - z)

            k = k + 1
            if k % 200 == 0 or (np.linalg.norm(w - w_o - z) <= eps or np.linalg.norm(np.sum(w) - 1) <= eps):
                ret = np.sum(w * x_t) / n
                log_ret = np.log(ret)
                cost = gamma * np.linalg.norm((w - w_o), ord=1)
                quad_error = np.linalg.norm(w - w_o)
                total = -log_ret + cost + 1 / (2 * eta) * quad_error

                print("iter {} return {} log_return {} cost {} quad_error {} total_value {}").format(k, ret, log_ret,
                                                                                                     cost, quad_error,
                                                                                                     total)

        return w

    @staticmethod
    def olmar(inputs):
        """
        paper: On-Line Portfolio Selection with Moving Average Reversion
        :param inputs: 算法输入
                x_pred: 相对价格预测向量，size = 1 * stocks
                w_o:    原来的组合权重，size = 1 * stocks
        :return: w(t+1)
        """
        # get inputs
        x_pred = inputs['x_pred']
        w_o = inputs['w_o']

        # algorithm constant
        eps = 10
        # omega = 5

        x_pred_avg = np.mean(x_pred)
        lam = max(0, (eps - np.sum(w_o * x_pred)) / (np.linalg.norm(x_pred - x_pred_avg) ** 2))
        w = w_o + lam * (x_pred - x_pred_avg)
        w = simplex_projection(w)

        return w

    @staticmethod
    def sspo(inputs):
        """
        paper: Short-term Sparse Portfolio Optimization Based on Alternating Direction Method of Multipliers
        :param inputs: 算法输入
                x_pred: 相对价格预测向量，size = 1 * stocks
                w_o:    原来的组合权重，size = 1 * stocks
        :return: w(t+1)
        """
        # get inputs
        x_pred = inputs['x_pred']
        w_o = inputs['w_o']

        # iteration constant
        eps = 1e-4
        max_iter = 10000

        # algorithm constant
        # omega = 5
        lam = 0.5
        gamma = 0.01
        eta = 0.005
        kxi = 500
        phi = -1.1 * np.log(x_pred) - 1

        # initialize
        n = len(w_o)
        w = np.copy(w_o)
        g = np.copy(w_o)
        rho = np.zeros(n)

        o = 1
        while np.abs(np.sum(w) - 1) > eps and o < max_iter:
            part_l = lam / gamma * np.ones((n, n)) + eta
            part_r = (lam / gamma) * g + (eta - rho) - phi
            w = np.dot(np.linalg.inv(part_l), part_r)
            part_l = np.sign(w)
            part_r = np.abs(w) - gamma * np.ones(n)
            for i in range(n):
                part_r[i] = part_r[i] if part_r[i] > 0 else 0
            g = part_l * part_r
            rho = rho + eta * (np.sum(w) - 1)

            o = o + 1
            if o % 200 == 0 or np.abs(np.sum(w) - 1) <= eps:
                R_t = 1.1 * np.log(x_pred) + 1
                sparsity = np.linalg.norm(w)
                total = -np.sum(w * R_t) + lam * np.linalg.norm(w, ord=1)
                print("iter {} R_t {} sparity {} total_value {}").format(o, R_t, sparsity, total)

        w = simplex_projection(w * kxi)
        return w
