from PySide6.QtCore import Qt, QEvent


keymap = {}
for key, value in vars(Qt).items():
    if isinstance(value, Qt.Key):
        keymap[value] = key.partition('_')[2]

def keyevent_to_string(event):
    sequence = []

    key = keymap.get(event.key(), event.text())
    if key not in sequence:
        sequence.append(key)
    return '+'.join(sequence)