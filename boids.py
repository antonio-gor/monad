"""
https://en.wikipedia.org/wiki/Boids
https://dl.acm.org/doi/pdf/10.1145/37401.37406
"""

import pygame
import random
import math
from typing import Any, List

FPS = 60
SCREEN_SIZE = 900
EDGE_WRAPPING = True
BOID_SIZE = 6
BOID_COLOR = pygame.Color("brown")
BOID_SPEED = 2
PROTECTED_RANGE = 40
SEPARATION_WEIGHT = 0.5

# pygame setup
pygame.init()
pygame.display.set_caption("Boids")
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
running = True
paused = False
next_step = False


class Boid:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = BOID_SPEED
        self.size = BOID_SIZE + random.uniform(-BOID_SIZE / 2, BOID_SIZE / 2)
        self.id = random.randint(0, 1000000)

    def boundary_perception(self) -> None:
        if EDGE_WRAPPING:
            if self.x < 0:
                self.x = SCREEN_SIZE
            elif self.x > SCREEN_SIZE:
                self.x = 0

            if self.y < 0:
                self.y = SCREEN_SIZE
            elif self.y > SCREEN_SIZE:
                self.y = 0
        else:
            if self.x < 0 or self.x > SCREEN_SIZE:
                self.direction = math.pi - self.direction
            if self.y < 0 or self.y > SCREEN_SIZE:
                self.direction = -self.direction

    def separation_perception(self, boids: List[Any]) -> None:
        total_repulsion_x = 0
        total_repulsion_y = 0
        for other_boid in boids:
            if other_boid.id == self.id:
                continue

            distance = math.hypot(other_boid.x - self.x, other_boid.y - self.y)
            if distance <= PROTECTED_RANGE:
                total_repulsion_x += self.x - other_boid.x
                total_repulsion_y += self.y - other_boid.y

        if total_repulsion_x or total_repulsion_y:
            repulsion_angle = math.atan2(total_repulsion_y, total_repulsion_x)
            self.direction = (
                1 - SEPARATION_WEIGHT
            ) * self.direction + SEPARATION_WEIGHT * repulsion_angle

    def move(self) -> None:
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed

    def draw(self, surface: pygame.Surface) -> None:
        front_point = (
            self.x + self.size * math.cos(self.direction),
            self.y + self.size * math.sin(self.direction),
        )
        left_point = (
            self.x + self.size * math.cos(self.direction + math.pi * 0.8),
            self.y + self.size * math.sin(self.direction + math.pi * 0.8),
        )
        right_point = (
            self.x + self.size * math.cos(self.direction - math.pi * 0.8),
            self.y + self.size * math.sin(self.direction - math.pi * 0.8),
        )
        pygame.draw.polygon(surface, BOID_COLOR, [front_point, left_point, right_point])


class Flock:
    def __init__(self, n: int = 50) -> None:
        self.boids = [
            Boid(random.randint(0, SCREEN_SIZE), random.randint(0, SCREEN_SIZE))
            for _ in range(n)
        ]

    def update(self, surface: pygame.Surface) -> None:
        for boid in self.boids:
            boid.boundary_perception()
            boid.separation_perception(self.boids)
            boid.move()
            boid.draw(surface)


flock = Flock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            if event.key == pygame.K_r:
                flock = Flock()

    screen.fill(pygame.Color(142, 220, 240))
    if not paused:
        flock.update(screen)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
