
def cast_value_to_str(value, verbose_ndigits):
    if type(value) == int:
        return f'{value}'
    else:
        return f'{value:.{verbose_ndigits}f}'


def prepare_text_msg(msg, verbose_ndigits=5, *args, **kwargs):
    msg = str(msg) if msg else ''
    for i, arg in enumerate(args):
        msg = f'{msg}, arg_{i}='
        msg += cast_value_to_str(arg, verbose_ndigits)
    for key, arg in kwargs.items():
        msg = f'{msg}, {key}='
        msg += cast_value_to_str(arg, verbose_ndigits)
    return msg
