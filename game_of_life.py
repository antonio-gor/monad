import pygame
import random

GRID_SIZE = 6
SCREEN_SIZE = 600
BUFFER = 0.9
CELL_SIZE = GRID_SIZE * BUFFER

# pygame setup
pygame.init()
pygame.display.set_caption("Conway's Game of Life")
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
running = True
dt = 0

ROWS = SCREEN_SIZE // GRID_SIZE
COLS = SCREEN_SIZE // GRID_SIZE
CENTER_X = int(ROWS / 2)
CENTER_Y = int(COLS / 2)


class Cell:
    def __init__(self, x: int, y: int, random_init: bool = True) -> None:
        self.x = x
        self.y = y
        self.alive = random.choice([True, False]) if random_init else False

    def live(self) -> None:
        self.alive = True

    def die(self) -> None:
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


class Grid:
    def __init__(self) -> None:
        self.cells = [[Cell(x, y) for y in range(COLS)] for x in range(ROWS)]

    def update(self) -> None:
        new_grid = [[Cell(x, y) for y in range(COLS)] for x in range(ROWS)]
        for row in range(ROWS):
            for col in range(COLS):
                new_grid[row][col].alive = self.cells[row][col].alive
                self.update_cell(new_grid[row][col])
        self.cells = new_grid

    def update_cell(self, cell: Cell) -> None:
        living_neighbors = self.find_living_neighbors(cell)
        if cell.is_alive() and living_neighbors < 2:
            cell.die()  # Underpopulation
        elif cell.is_alive() and living_neighbors in [2, 3]:
            cell.live()  # Stays alive
        elif cell.is_alive() and living_neighbors > 3:
            cell.die()  # Overpopulation
        elif not cell.is_alive() and living_neighbors == 3:
            cell.live()  # Reproduction

    def find_living_neighbors(self, cell: Cell) -> int:
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # skip current cell
                nx, ny = cell.x + dx, cell.y + dy
                if 0 <= nx < ROWS and 0 <= ny < COLS and self.cells[nx][ny].is_alive():
                    neighbors += 1  # found living cell
        return neighbors

    def draw(self) -> None:
        for row in range(ROWS):
            for col in range(COLS):
                color = "white" if self.cells[row][col].is_alive() else "black"
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(col * GRID_SIZE, row * GRID_SIZE, CELL_SIZE, CELL_SIZE),
                )


grid = Grid()

# Glider
grid.cells[CENTER_Y - 1][CENTER_X].alive = True
grid.cells[CENTER_Y][CENTER_X + 1].alive = True
grid.cells[CENTER_Y + 1][CENTER_X + 1].alive = True
grid.cells[CENTER_Y + 1][CENTER_X].alive = True
grid.cells[CENTER_Y + 1][CENTER_X - 1].alive = True

while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update and draw the grid
    grid.update()
    screen.fill("black")
    grid.draw()

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(10) / 1000

pygame.quit()
