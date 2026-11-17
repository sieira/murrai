from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from murrai.presentation.chat.repl import ABOUT_TEXT, AboutDialog


def mouse(kind):
    return MouseEvent(Point(x=0, y=0), kind, MouseButton.LEFT, frozenset())


def about(closed):
    return AboutDialog(
        on_close=lambda: closed.append(1),
        focus=lambda control: None,
    )


def test_the_about_popover_centers_itself_without_a_position():
    dialog = about([])

    assert dialog.float.left is None
    assert dialog.float.top is None


def test_the_popover_is_exactly_as_wide_as_its_text_and_its_frame():
    dialog = about([])
    dialog.open()

    width = dialog.container.preferred_width(100).preferred

    assert width == len(ABOUT_TEXT) + 2


def test_the_popover_shadow_is_two_cells_wide_and_one_row_deep():
    dialog = about([])

    widths = [f.width for f in dialog.shadow.container.floats if f.width]
    heights = [f.height for f in dialog.shadow.container.floats if f.height]

    assert widths == [2]
    assert heights == [1]


def test_the_about_popover_never_shows_the_cursor():
    dialog = about([])

    assert dialog.cross.content.show_cursor is False
    assert dialog.text.content.show_cursor is False


def test_the_cross_closes_on_release_not_on_hover():
    closed = []
    dialog = about(closed)
    dialog.open()

    cross = next(f for f in dialog.cross_fragments() if f[1].strip() == "×")  # noqa: RUF001
    cross[2](mouse(MouseEventType.MOUSE_MOVE))
    cross[2](mouse(MouseEventType.MOUSE_DOWN))
    assert dialog.visible

    cross[2](mouse(MouseEventType.MOUSE_UP))
    assert not dialog.visible
    assert closed == [1]


def test_the_cross_sits_on_the_right_edge_of_the_popover():
    dialog = about([])
    dialog.open()

    screen = Screen(initial_width=30, initial_height=10)
    dialog.container.write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(0, 0, 30, 8),
        "",
        False,
        0,
    )
    places = [
        (x, y)
        for y in range(10)
        for x in range(30)
        if screen.data_buffer[y][x].char == "×"  # noqa: RUF001
    ]

    assert places
    assert all(x > 20 for x, _ in places)
