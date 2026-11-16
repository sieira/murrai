from collections.abc import Callable
from dataclasses import dataclass

GAP = 2


@dataclass
class MenuItem:
    name: str
    description: str
    action: Callable


@dataclass
class Menu:
    title: str
    items: tuple = ()


class MenuBar:
    def __init__(self, menus):
        self.menus = tuple(menus)
        self.active = False
        self.open_index = None
        self.item_index = 0

    @property
    def open_menu(self):
        if self.open_index is None:
            return None
        return self.menus[self.open_index]

    @property
    def current(self):
        menu = self.open_menu
        if menu is None or not menu.items:
            return None
        return menu.items[self.item_index]

    def open(self, index):
        self.active = True
        self.open_index = index % len(self.menus)
        self.item_index = 0

    def close(self):
        self.active = False
        self.open_index = None
        self.item_index = 0

    def move_menu(self, step):
        if self.open_index is None:
            return
        self.open(self.open_index + step)

    def move_item(self, step):
        menu = self.open_menu
        if menu is None or not menu.items:
            return
        self.item_index = (self.item_index + step) % len(menu.items)

    def pick(self):
        if self.open_index is None:
            return None
        item = self.current
        self.close()
        return item

    def jump(self, letter):
        menu = self.open_menu
        if menu is not None:
            for item in menu.items:
                if item.name[:1].lower() == letter.lower():
                    self.close()
                    return item
        for index, menu in enumerate(self.menus):
            if menu.title[:1].lower() == letter.lower():
                self.open(index)
                return None
        return None


def menu_columns(menubar):
    columns = []
    column = 0
    for menu in menubar.menus:
        columns.append(column)
        column += len(menu.title) + GAP
    return columns


def _toggle(menubar, index):
    def handler(event):
        if menubar.open_index == index:
            menubar.close()
        else:
            menubar.open(index)

    return handler


def bar_fragments(menubar):
    fragments = []
    for index, menu in enumerate(menubar.menus):
        if index:
            fragments.append(("", " " * GAP))
        style = "class:menu-selected" if index == menubar.open_index else "class:menu"
        fragments.append(
            (style + " underline", menu.title[:1], _toggle(menubar, index)),
        )
        if menu.title[1:]:
            fragments.append((style, menu.title[1:], _toggle(menubar, index)))
    return fragments


def _picker(menubar):
    def handler(event):
        item = menubar.pick()
        if item is not None:
            item.action()

    return handler


def menu_fragments(menubar):
    menu = menubar.open_menu
    if menu is None:
        return []
    handler = _picker(menubar)
    fragments = []
    for index, item in enumerate(menu.items):
        style = "class:menu-selected" if index == menubar.item_index else "class:menu"
        fragments.append((style, item.name, handler))
        fragments.append(("", "\n"))
    return fragments


def description_fragments(menubar):
    item = menubar.current
    if item is None:
        return []
    return [("class:hint", item.description)]
