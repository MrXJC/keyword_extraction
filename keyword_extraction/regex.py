# -*- coding: utf-8 -*-

import re
EMAIL_RE = r'[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+){0,4}@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+){0,4}'


def delete_email(text):
    """Docstring for delete_email.

    :arg1: text
    :returns: text

    """
    return re.sub(EMAIL_RE, '', text)
