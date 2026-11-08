def fib(n):
    """返回第 n 个斐波那契数（F0=0, F1=1）。n >= 0。"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
