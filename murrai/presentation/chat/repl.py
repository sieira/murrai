def render_dice(history, dice, rolled, top=8):
    label = history if len(history) <= 10 else "…" + history[-9:]
    lines = [f"after '{label}', the die:"]
    lines.extend(f"  {c} {p:.1%}" for c, p in dice[:top])
    if len(dice) > top:
        lines.append(f"  ... {len(dice) - top} more")
    lines.append(f"rolled '{rolled}'")
    return "\n".join(lines) + "\n"


def run_chat(session, read, write):
    write("chat: one char per turn; ctrl-d or ctrl-c quits\n")
    while True:
        write("char> ")
        try:
            line = read()
        except KeyboardInterrupt:
            write("bye\n")
            return
        if line is None or line == "":
            write("bye\n")
            return
        char = line.strip()[:1]
        if len(char) != 1:
            write("one character per turn\n")
            continue
        try:
            turn = session.turn(char)
            write(render_dice(turn.history, turn.dice, turn.rolled))
        except ValueError:
            write("unknown char for this corpus\n")
        except KeyboardInterrupt:
            write("\nbye\n")
            return
