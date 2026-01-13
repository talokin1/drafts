\bтранзит\w*\b.*\b(mpos|pos)\b|\b(mpos|pos)\b.*\bтранзит\w*\b
if RE_ACQ_TRANSIT.search(text):
    return True

