import math
import pygame
from random import randint, uniform
from typing import List
from uuid import uuid4

SCREEN_SIZE_X = 1280
SCREEN_SIZE_Y = 720
PARTICLE_COUNT = 200
VELOCITY_SCALER = 2
COLOR_MODE = "velocity"  # by mass or velocity
MASS_COLORS = {3: "red", 4: "green", 5: "blue", 6: "purple"}


class Particle:
    def __init__(self, position: List, velocity: List, mass: float) -> None:
        self.position = position
        self.velocity = velocity
        self.speed = self.get_speed()
        self.mass = mass
        self.color = MASS_COLORS[self.mass]
        self.id = uuid4()

    def get_speed(self) -> float:
        return math.hypot(self.velocity[1], self.velocity[0])

    def move(self):
        if self.position[0] < 0 or self.position[0] > SCREEN_SIZE_X:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] < 0 or self.position[1] > SCREEN_SIZE_Y:
            self.velocity[1] = -self.velocity[1]

        self.position[0] = self.position[0] + self.velocity[0] * VELOCITY_SCALER
        self.position[1] = self.position[1] + self.velocity[1] * VELOCITY_SCALER
        self.speed = self.get_speed()

    def draw(self, surface: pygame.Surface, top_speed: float) -> None:
        color = self.color
        if COLOR_MODE == "velocity":
            speed = math.hypot(self.velocity[1], self.velocity[0])
            speed_normalized = speed / top_speed
            speed_color = min(255, int(255 * speed_normalized))
            color = pygame.Color(speed_color, 0, 255 - speed_color)
        pygame.draw.circle(
            surface=surface,
            color=color,
            center=self.position,
            radius=self.mass,
        )


class System:
    def __init__(self, size: int = 50) -> None:
        self.particles = []
        for _ in range(size):
            self.create_particle()
        self.top_speed = self.get_top_speed()

    def create_particle(self):
        particle = Particle(
            position=[randint(0, SCREEN_SIZE_X), randint(0, SCREEN_SIZE_Y)],
            velocity=[uniform(-1, 1), uniform(-1, 1)],
            mass=randint(min(MASS_COLORS.keys()), max(MASS_COLORS.keys())),
        )
        self.particles.append(particle)

    def get_top_speed(self) -> float:
        top_speed = 0.01
        for particle in self.particles:
            if particle.speed > top_speed:
                top_speed = particle.speed
        return top_speed

    def update(self, surface: pygame.Surface) -> None:
        for particle in self.particles:
            particle.move()
            self.top_speed = self.get_top_speed()
            particle.draw(surface, self.top_speed)


pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE_X, SCREEN_SIZE_Y))
clock = pygame.time.Clock()
running = True

system = System(size=PARTICLE_COUNT)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                system = System(size=PARTICLE_COUNT)

    screen.fill("black")

    system.update(screen)

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
