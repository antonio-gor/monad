import copy
import pygame
import random

# Main settings
FPS = 60  # Main FPS for updates and user interaction
RENDER_FPS = 20  # FPS for rendering
GRID_SIZE = 6
SCREEN_SIZE = 800
BUFFER = 0.9
CELL_SIZE = GRID_SIZE * BUFFER

# pygame setup
pygame.init()
pygame.display.set_caption("Conway's Game of Life")
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
running = True
paused = False
next_step = False
dt = 0
frame_count = 0  # Counter to track frame progression
render_interval = FPS // RENDER_FPS  # Calculate how often to render based on FPS

ROWS = SCREEN_SIZE // GRID_SIZE
COLS = SCREEN_SIZE // GRID_SIZE
CENTER_X = int(ROWS / 2)
CENTER_Y = int(COLS / 2)


class Cell:
    def __init__(self, x: int, y: int, randomize: bool = False) -> None:
        self.x = x
        self.y = y
        self.alive = random.choice([True, False]) if randomize else False
        self.age_alive = 0

    def live(self) -> None:
        self.alive = True
        self.age_alive += 35

    def die(self) -> None:
        self.alive = False
        self.age_alive = max(0, self.age_alive - 8)

    def is_alive(self) -> bool:
        return self.alive


class Grid:
    def __init__(self, randomize: bool = False) -> None:
        self.randomize = randomize
        self.cells = [[Cell(x, y, randomize) for y in range(COLS)] for x in range(ROWS)]

    def update(self) -> None:
        new_grid = [
            [copy.deepcopy(self.cells[x][y]) for y in range(COLS)] for x in range(ROWS)
        ]
        for row in range(ROWS):
            for col in range(COLS):
                self.update_cell(new_grid[row][col])
        self.cells = new_grid

    def update_cell(self, cell: Cell) -> None:
        living_neighbors = self.find_living_neighbors(cell)
        if cell.is_alive() and living_neighbors < 2:
            cell.die()  # Underpopulation
        elif cell.is_alive() and living_neighbors in [2, 3]:
            cell.live()  # Stays alive
        elif not cell.is_alive() and living_neighbors == 3:
            cell.live()  # Reproduction
        elif cell.is_alive() and living_neighbors > 3:
            cell.die()  # Overpopulation

        else:
            cell.die()

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

    def toggle_cell(self, x, y):
        cell = self.cells[x][y]
        if cell.is_alive():
            cell.age_alive = 0
            cell.die()
        else:
            cell.age_alive = 100
            cell.live()

    def draw(self) -> None:
        for row in range(ROWS):
            for col in range(COLS):
                cell = self.cells[row][col]
                color = pygame.Color(10, 10, 10)
                if cell.is_alive():  # brighter white the longer they live
                    color_value = min(cell.age_alive, 255)
                    color = pygame.Color(color_value, color_value, color_value)
                elif cell.age_alive > 0:
                    color_value = min(cell.age_alive, 255)
                    color = pygame.Color(color_value, 0, 0)  # red fading over time
                    # color = pygame.Color(color_value, color_value, color_value)
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(col * GRID_SIZE, row * GRID_SIZE, CELL_SIZE, CELL_SIZE),
                )


# Start simulation
grid = Grid(randomize=True)

while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            if event.key == pygame.K_r:
                grid = Grid(randomize=True)
            if event.key == pygame.K_c:
                grid = Grid(randomize=False)
                paused = True
            if event.key == pygame.K_n:
                next_step = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            col = mouse_pos[0] // GRID_SIZE
            row = mouse_pos[1] // GRID_SIZE
            if 0 <= col < COLS and 0 <= row < ROWS:
                grid.toggle_cell(row, col)

    # Render the grid every `render_interval` frames
    frame_count += 1
    if frame_count >= render_interval:
        if not paused or (paused and next_step):
            grid.update()
            next_step = False
        screen.fill(pygame.Color(42, 42, 42))
        grid.draw()
        pygame.display.flip()
        frame_count = 0

    # Control FPS
    dt = clock.tick(FPS) / 1000

pygame.quit()
