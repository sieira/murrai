def render_dice(prev_char, dice, rolled, top=8):
    lines = [f"after '{prev_char}', the die:"]
    lines.extend(f"  {c} {p:.1%}" for c, p in dice[:top])
    if len(dice) > top:
        lines.append(f"  ... {len(dice) - top} more")
    lines.append(f"rolled '{rolled}'")
    return "\n".join(lines) + "\n"


def run_chat(session, read, write):
    write("bigram chat: one char per turn; empty line or q quits\n")
    while True:
        write("char> ")
        line = read()
        if line is None or not line.strip():
            write("bye\n")
            return
        if line.strip().lower() in ("q", "quit"):
            write("bye\n")
            return
        char = line.strip()[:1]
        if len(char) != 1:
            write("one character per turn\n")
            continue
        try:
            write(render_dice(char, session.dice(char), session.roll(char)))
        except ValueError:
            write("unknown char for this corpus\n")
