"""
Left-click  : reveal
Right-click : flag / unflag
"""

import tkinter as tk
from tkinter import messagebox
import random

PRESETS = {
    "Beginner":     (9,  9,  10),
    "Intermediate": (16, 16, 40),
    "Expert":       (16, 30, 99),
}

CLUE_COLORS = {
    1: "#1565C0", 2: "#2E7D32", 3: "#C62828",
    4: "#283593", 5: "#6D1A36", 6: "#00695C",
    7: "#212121", 8: "#757575",
}

class Game:
    def __init__(self, rows, cols, n_mines):
        self.rows    = rows
        self.cols    = cols
        self.n_mines = n_mines
        self.mines   = [[False]*cols for _ in range(rows)]
        self.clues   = [[0]*cols     for _ in range(rows)]
        self.shown   = [[False]*cols for _ in range(rows)]
        self.flagged = [[False]*cols for _ in range(rows)]
        self.started = False
        self.over    = False
        self.won     = False

    def _in(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _nb(self, r, c):
        return [(r+dr, c+dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and self._in(r+dr, c+dc)]

    def _setup(self, sr, sc):
        safe = {(sr+dr, sc+dc)
                for dr in (-1,0,1) for dc in (-1,0,1)
                if self._in(sr+dr, sc+dc)}
        safe.add((sr, sc))
        pool = [(r, c) for r in range(self.rows)
                       for c in range(self.cols)
                       if (r, c) not in safe]
        for r, c in random.sample(pool, min(self.n_mines, len(pool))):
            self.mines[r][c] = True
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.mines[r][c]:
                    self.clues[r][c] = sum(
                        self.mines[nr][nc] for nr, nc in self._nb(r, c))
        self.started = True

    def reveal(self, r, c):
        if self.over or self.flagged[r][c] or self.shown[r][c]:
            return
        if not self.started:
            self._setup(r, c)
        if self.mines[r][c]:
            self.shown[r][c] = True
            self.over = True
            return
        self._flood(r, c)
        if all(self.shown[r][c] or self.mines[r][c]
               for r in range(self.rows)
               for c in range(self.cols)):
            self.won = self.over = True

    def _flood(self, r, c):
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if not self._in(r, c) or self.shown[r][c] or self.flagged[r][c]:
                continue
            self.shown[r][c] = True
            if self.clues[r][c] == 0:
                stack.extend(self._nb(r, c))

    def flag(self, r, c):
        if not self.over and not self.shown[r][c]:
            self.flagged[r][c] = not self.flagged[r][c]

    def flags_placed(self):
        return sum(self.flagged[r][c]
                   for r in range(self.rows) for c in range(self.cols))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minesweeper")
        self.resizable(False, False)
        self.preset = tk.StringVar(value="Intermediate")
        self.btns = []
        self.game = None
        self.grid_frame = None

        self._build_top()
        self.bind("<F2>", lambda _: self.new_game())
        self.new_game()

    def _build_top(self):
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        for p in PRESETS:
            tk.Radiobutton(top, text=p, variable=self.preset,
                value=p, command=self.new_game).pack(side=tk.LEFT)
            self.mine_lbl = tk.Label(top, text="", width=12)
            self.mine_lbl.pack(side=tk.RIGHT)
            tk.Button(top, text="New Game (F2)",
                  command=self.new_game).pack(side=tk.RIGHT, padx=6)

    def new_game(self):
        #destroy the old grid
        if self.grid_frame is not None:
            self.grid_frame.destroy()

        self.grid_frame = tk.Frame(self, bd=2, relief=tk.SUNKEN)
        self.grid_frame.pack(padx=6, pady=6)

        self.btns = []
        rows, cols, mines = PRESETS[self.preset.get()]
        self.game = Game(rows, cols, mines)

        for r in range(rows):
            row_btns = []
            for c in range(cols):
                #if statement here to implement the different colors
                b = tk.Label(
                    self.grid_frame,
                    width=2,
                    height=1,
                    font=("Courier", 11, "bold"),
                    relief=tk.RAISED,
                    bd=2,
                    bg="#BBBBBB",
                    text="",
                )

                b.grid(row=r, column=c, padx=0, pady=0)
                b.bind("<Button-1>", lambda e, r=r, c=c: self._left(r, c))
                b.bind("<Button-2>", lambda e, r=r, c=c: self._right(r, c))
                b.bind("<Button-3>", lambda e, r=r, c=c: self._right(r, c))
                row_btns.append(b)
            self.btns.append(row_btns)

        self._update_label()

    def _left(self, r, c):
        if self.game.over:
            return
        self.game.reveal(r, c)
        self._redraw()
        if self.game.over:
            self._end()

    def _right(self, r, c):
        if self.game.over:
            return
        self.game.flag(r, c)
        self._redraw()

    def _redraw(self):
        g = self.game
        for r in range(g.rows):
            for c in range(g.cols):
                b = self.btns[r][c]
                if g.shown[r][c]:
                    if g.mines[r][c]:
                        b.config(text="*", bg="#E53935",
                                 fg="white", relief=tk.SUNKEN)
                    else:
                        n = g.clues[r][c]
                        b.config(
                            text=str(n) if n else "",
                            fg=CLUE_COLORS.get(n, "#000000"),
                            relief=tk.SUNKEN,
                        )
                elif g.flagged[r][c]:
                    b.config(text="F", bg="#BBBBBB",
                             fg="#E53935", relief=tk.RAISED)
                else:
                    b.config(text="", bg="#BBBBBB",
                             fg="black", relief=tk.RAISED)
        self._update_label()

    def _update_label(self):
        self.mine_lbl.config(
            text=f"Mines: {self.game.n_mines - self.game.flags_placed()}")

    def _end(self):
        g = self.game
        for r in range(g.rows):
            for c in range(g.cols):
                if g.mines[r][c]:
                    g.shown[r][c] = True
        self._redraw()
        if g.won:
            messagebox.showinfo("Minesweeper", "Win!")
        else:
            messagebox.showinfo("Minesweeper", "BOOM!")


if __name__ == "__main__":
    App().mainloop()