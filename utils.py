import numpy as np

def s_entropy(freq_list):
    """
    Compute Shannon entropy of a given frequency list (probability distribution).

    Parameters:
        freq_list (list or np.ndarray): List of non-zero probabilities.

    Returns:
        float: Shannon entropy.
    """
    freq_list = [element for element in freq_list if element != 0]
    sh_entropy = -sum(freq * np.log(freq) for freq in freq_list)
    return sh_entropy


def _hash(x):
    """
    Recursively compute a unique hash index for each permutation of ordinal patterns.

    Parameters:
        x (np.ndarray): Matrix of ordinal pattern permutations.

    Returns:
        np.ndarray: Array of hashed indices for each permutation.
    """
    m, n = x.shape
    if n == 1:
        return np.zeros(m)
    return np.sum(np.apply_along_axis(lambda y: y < x[:, 0], 0, x), axis=1) * np.math.factorial(n-1) + _hash(x[:, 1:])


def ordinal_patterns(ts, embdim, embdelay):
    """
    Extract ordinal patterns from a time series.

    Parameters:
        ts (array-like): Time series data.
        embdim (int): Embedding dimension.
        embdelay (int): Embedding delay.

    Returns:
        list: Frequency of each unique ordinal pattern (non-zero only).
    """
    m, t = embdim, embdelay
    x = np.array(ts)

    tmp = np.zeros((x.shape[0], m))
    for i in range(m):
        tmp[:, i] = np.roll(x, i*t)
    partition = tmp[(t*(m-1)):, :]

    permutation = np.argsort(partition)
    idx = _hash(permutation)

    counts = np.zeros(np.math.factorial(m))
    for i in range(len(counts)):
        counts[i] = (idx == i).sum()

    return list(counts[counts != 0].astype(int))


def p_entropy(op):
    """
    Compute normalized permutation entropy from ordinal patterns.

    Parameters:
        op (list): List of ordinal pattern frequencies.

    Returns:
        float: Normalized permutation entropy.
    """
    max_entropy = np.log(len(op))
    p = np.array(op) / sum(op)
    return s_entropy(p) / max_entropy


def complexity(op):
    """
    Compute statistical complexity (based on Jensen-Shannon divergence).

    Parameters:
        op (list): Ordinal pattern frequencies.

    Returns:
        float: Statistical complexity.
    """
    pe = p_entropy(op)
    length = len(op)

    # Constants for Jensen-Shannon divergence baseline
    uniform = 1 / length
    constant1 = (0.5 + (0.5 / length)) * np.log(0.5 + (0.5 / length))
    constant2 = (0.5 / length) * np.log(0.5 / length) * (length - 1)
    constant3 = 0.5 * np.log(length)
    Q_o = -1 / (constant1 + constant2 + constant3)

    temp_op_prob = np.array(op) / sum(op)
    temp_op_prob2 = 0.5 * temp_op_prob + 0.5 * uniform

    # Jensen-Shannon divergence
    JSdivergence = (
        s_entropy(temp_op_prob2) -
        0.5 * s_entropy(temp_op_prob) -
        0.5 * np.log(length)
    )

    return Q_o * JSdivergence * pe


def weighted_ordinal_patterns(ts, embdim, embdelay):
    """
    Compute weighted ordinal pattern frequencies where weights are the variance within the embedding.

    Parameters:
        ts (array-like): Time series data.
        embdim (int): Embedding dimension.
        embdelay (int): Embedding delay.

    Returns:
        list: Weighted frequencies for each ordinal pattern (non-zero only).
    """
    m, t = embdim, embdelay
    x = np.array(ts)

    tmp = np.zeros((x.shape[0], m))
    for i in range(m):
        tmp[:, i] = np.roll(x, i * t)
    partition = tmp[(t * (m - 1)):, :]

    xm = np.mean(partition, axis=1)
    weight = np.mean((partition - xm[:, None])**2, axis=1)

    permutation = np.argsort(partition)
    idx = _hash(permutation)

    counts = np.zeros(np.math.factorial(m))
    for i in range(len(counts)):
        counts[i] = sum(weight[i == idx])

    return list(counts[counts != 0])
