import math
import pygame
from random import randint, uniform
from typing import List, Tuple
from uuid import uuid4

SCREEN_SIZE_X = 1280
SCREEN_SIZE_Y = 720
FPS = 30
PARTICLE_COUNT = 450
PARTICLE_SIZE = 2
VELOCITY_SCALER = 1
SPEED_LIMIT = 6
DRAW_VECTORS = False
INIT_STATIC = True
COLOR_MODE = "type"  # by "type" or "velocity"
INTERACTION_RADIUS = 100
REPULSION_RADIUS = 20
REPULSION_SCALAR = 1
ATTRACTION_SCALAR = 4
TYPE_COLORS = {
    0: pygame.Color("cyan"),
    1: pygame.Color("orange"),
    2: pygame.Color("magenta"),
    3: pygame.Color("green"),
    4: pygame.Color("red"),
}
TYPE_INTERACTIONS = [
    [1, 1, 0.2, 0.2, 0],
    [1, 1, 0.2, 0.2, 0],
    [0.2, 0.2, -1, 2, 0],
    [0.2, 0.2, 2, -1, 0],
    [0.5, 0.5, -0.2, -0.2, -0.5]
]
# TYPE_COLORS = {
#     0: pygame.Color("green"),
#     1: pygame.Color("cyan"),
#     2: pygame.Color("yellow"),
#     3: pygame.Color("magenta"),
#     4: pygame.Color("orange"),
# }
# TYPE_INTERACTIONS = [
#     [1, 1, 1, 1, 1],
#     [1, -1, 0, 0, 0],
#     [1, 0, -1, 0, 0],
#     [1, 0, 0, -1, 0],
#     [1, 0, 0, 0, -1],
# ]


class SpatialGrid:
    """Reference: http://gameprogrammingpatterns.com/spatial-partition.html"""

    def __init__(self, cell_size: int) -> None:
        self.cell_size = cell_size
        self.cells = {}

    def add_particle(self, particle: "Particle") -> None:
        cell_coords = self.get_cell_coords(particle.position)
        self.cells.setdefault(cell_coords, []).append(particle)

    def get_cell_coords(self, position: List[int]) -> Tuple[int, int]:
        return (
            int(position[0] // self.cell_size),
            int(position[1] // self.cell_size),
        )

    def get_neighboring_particles(self, particle: "Particle") -> List:
        neighboring_particles = []
        cell_x, cell_y = self.get_cell_coords(particle.position)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.cells:
                    neighboring_particles.extend(self.cells[neighbor_cell])
        return neighboring_particles


class Particle:
    def __init__(self, position: List, velocity: List, type: int) -> None:
        self.position = position
        self.velocity = velocity
        self.speed = self.get_speed()
        self.type = type
        self.color = TYPE_COLORS[self.type]
        self.id = uuid4()

    def get_speed(self) -> float:
        return math.hypot(self.velocity[1], self.velocity[0])

    def move(self, particles: List["Particle"]) -> None:
        self.boundary_detection()
        self.neighbor_interactions(particles)
        self.cap_velocity()
        self.update_position()

    def cap_velocity(self) -> None:
        speed = self.get_speed()
        if speed > SPEED_LIMIT:
            scaling_factor = SPEED_LIMIT / speed
            self.velocity[0] *= scaling_factor
            self.velocity[1] *= scaling_factor

    def boundary_detection(self) -> None:
        if self.position[0] < 0:
            self.position[0] = 0
            self.velocity[0] = -self.velocity[0]
        if self.position[0] > SCREEN_SIZE_X:
            self.position[0] = SCREEN_SIZE_X
            self.velocity[0] = -self.velocity[0]
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = -self.velocity[1]
        if self.position[1] > SCREEN_SIZE_Y:
            self.position[1] = SCREEN_SIZE_Y
            self.velocity[1] = -self.velocity[1]

    def neighbor_interactions(self, grid: SpatialGrid) -> None:
        neighboring_particles = grid.get_neighboring_particles(self)
        for other in neighboring_particles:
            if self.id == other.id:
                continue

            dx = other.position[0] - self.position[0]
            dy = other.position[1] - self.position[1]
            distance = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            force_magnitude = 0

            # repulsive force
            if distance <= REPULSION_RADIUS:
                force_magnitude = distance / REPULSION_RADIUS - 1
                force_magnitude *= REPULSION_SCALAR
            # attractive force
            elif REPULSION_RADIUS < distance < INTERACTION_RADIUS:
                force_magnitude = (
                    distance / INTERACTION_RADIUS
                ) * ATTRACTION_SCALAR - 1
                attraction_factor = TYPE_INTERACTIONS[self.type][other.type]
                force_magnitude *= attraction_factor

            self.velocity[0] += force_magnitude * math.cos(angle)
            self.velocity[1] += force_magnitude * math.sin(angle)

    def update_position(self) -> None:
        self.position[0] += self.velocity[0] * VELOCITY_SCALER
        self.position[1] += self.velocity[1] * VELOCITY_SCALER
        self.speed = self.get_speed()

    def draw(self, surface: pygame.Surface) -> None:
        if COLOR_MODE == "velocity":
            speed_normalized = self.speed / SPEED_LIMIT
            speed_color = min(255, int(255 * speed_normalized))
            color = pygame.Color(255, 255 - speed_color, 255 - speed_color)
        else:
            color = pygame.Color(self.color)

        pygame.draw.circle(
            surface=surface,
            color=color,
            center=self.position,
            radius=PARTICLE_SIZE,
        )

        if DRAW_VECTORS:
            end_position = (
                self.position[0] + self.velocity[0] * 8,
                self.position[1] + self.velocity[1] * 8,
            )
            pygame.draw.line(surface, "red", self.position, end_position)


class System:
    def __init__(self, size: int = 50, init_static: bool = False) -> None:
        self.particles = []
        self.init_static = init_static
        self.grid = SpatialGrid(cell_size=INTERACTION_RADIUS)
        for _ in range(size):
            self.create_particle()

    def create_particle(self, position: List = None):
        random_position = [randint(0, SCREEN_SIZE_X), randint(0, SCREEN_SIZE_Y)]
        random_velocity = [uniform(-1, 1), uniform(-1, 1)]
        position = position if position else random_position
        velocity = [0, 0] if self.init_static else random_velocity
        particle = Particle(
            position=position,
            velocity=velocity,
            type=randint(min(TYPE_COLORS.keys()), max(TYPE_COLORS.keys())),
        )
        self.grid.add_particle(particle)
        self.particles.append(particle)

    def update(self, surface: pygame.Surface) -> None:
        self.grid = SpatialGrid(cell_size=INTERACTION_RADIUS)
        for particle in self.particles:
            self.grid.add_particle(particle)
        for particle in self.particles:
            particle.move(self.grid)
            particle.draw(surface)


pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE_X, SCREEN_SIZE_Y))
clock = pygame.time.Clock()
running = True

system = System(size=PARTICLE_COUNT, init_static=INIT_STATIC)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                system = System(size=PARTICLE_COUNT, init_static=INIT_STATIC)
            if event.key == pygame.K_c:
                COLOR_MODE = "velocity" if COLOR_MODE == "type" else "type"
            if event.key == pygame.K_v:
                DRAW_VECTORS = not DRAW_VECTORS
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_position = list(pygame.mouse.get_pos())
            system.create_particle(position=mouse_position)

    screen.fill("black")

    system.update(screen)

    pygame.display.flip()

    clock.tick(FPS)
    print(f"fps: {clock.get_fps()}", end="\r")

pygame.quit()
