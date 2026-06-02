"""
Approximate multiplier Approxx 10 (8-bit × 8-bit → 16-bit).

Source: Sabetzadeh, Moaiyeri, Ahmadinejad, IEEE TCAS-II vol.70 no.2, Feb. 2023.
"""


def exact_multiply(a: int, b: int) -> int:
    return (a & 0xFF) * (b & 0xFF)


def approx10_multiply(a: int, b: int) -> int:
    a &= 0xFF
    b &= 0xFF

    # Exact part: sum partial products from columns 8–14
    upper = 0
    for i in range(8):
        ai = (a >> i) & 1
        if ai == 0:
            continue
        for j in range(max(8 - i, 0), 8):
            if (b >> j) & 1:
                upper += 1 << (i + j)

    # ECM: two OR-4 gates on column-7 partial products
    or1 = (
        (((a >> 0) & 1) & ((b >> 7) & 1))
        | (((a >> 1) & 1) & ((b >> 6) & 1))
        | (((a >> 2) & 1) & ((b >> 5) & 1))
        | (((a >> 3) & 1) & ((b >> 4) & 1))
    )
    or2 = (
        (((a >> 4) & 1) & ((b >> 3) & 1))
        | (((a >> 5) & 1) & ((b >> 2) & 1))
        | (((a >> 6) & 1) & ((b >> 1) & 1))
        | (((a >> 7) & 1) & ((b >> 0) & 1))
    )

    # Constant-truncated region (bits 7:0 = 6) + ECM carry injection
    return (upper + 6 + (or1 + or2) * 256) & 0xFFFF


def approx10_multiply_no_ecm(a: int, b: int) -> int:
    """Approxx 10 without ECM — reference variant for ecm_comparison.png."""
    a &= 0xFF
    b &= 0xFF
    upper = 0
    for i in range(8):
        ai = (a >> i) & 1
        if ai == 0:
            continue
        for j in range(max(8 - i, 0), 8):
            if (b >> j) & 1:
                upper += 1 << (i + j)
    return (upper + 6) & 0xFFFF
