"""
https://en.wikipedia.org/wiki/Boids
https://dl.acm.org/doi/pdf/10.1145/37401.37406
"""

import pygame
import random
import math
from typing import List

FPS = 60
SCREEN_SIZE = 900
EDGE_WRAPPING = False
BOID_SIZE = 3
BOID_COLOR = pygame.Color("brown")
BOID_SPEED = 2
BOID_MAX_SPEED = 3
PROTECTED_RANGE = 25
VISUAL_RANGE = 150
SEPARATION_WEIGHT = 0.2
ALIGNMENT_WEIGHT = 0.1
COHESION_WEIGHT = 0.1

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
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.angle = math.atan2(self.vy, self.vx)
        self.size = BOID_SIZE + random.uniform(-BOID_SIZE / 4, BOID_SIZE / 4)
        self.id = random.randint(0, 1000000)
        self.selected = False

    def move(self) -> None:
        speed = math.hypot(self.vx, self.vy)
        if speed > BOID_MAX_SPEED:
            self.vx = (self.vx / speed) * BOID_MAX_SPEED
            self.vy = (self.vy / speed) * BOID_MAX_SPEED
        self.x += self.vx * BOID_SPEED
        self.y += self.vy * BOID_SPEED
        self.angle = math.atan2(self.vy, self.vx)

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
                self.angle = math.pi - self.angle
            if self.y < 0 or self.y > SCREEN_SIZE:
                self.angle = -self.angle

    def separation(self, boids: List["Boid"]) -> None:
        neighbors = 0
        neighbors_dx = 0
        neighbors_dy = 0

        for other in boids:
            if other.id == self.id:
                continue

            dx = other.x - self.x
            dy = other.y - self.y
            distance = math.hypot(dx, dy)
            if 0 < distance < PROTECTED_RANGE:
                neighbors += 1
                neighbors_dx += dx / distance
                neighbors_dy += dy / distance

        if neighbors:
            neighbors_dx_avg = neighbors_dx / neighbors
            neighbors_dy_avg = neighbors_dy / neighbors
            self.vx -= neighbors_dx_avg * SEPARATION_WEIGHT
            self.vy -= neighbors_dy_avg * SEPARATION_WEIGHT

    def alignment(self, boids: List["Boid"]) -> None:
        neighbors = 0
        neighbors_vx = 0
        neighbors_vy = 0

        for other in boids:
            if other.id == self.id:
                continue

            dx = other.x - self.x
            dy = other.y - self.y
            distance = math.hypot(dx, dy)
            if PROTECTED_RANGE <= distance <= VISUAL_RANGE:
                neighbors += 1
                neighbors_vx += other.vx
                neighbors_vy += other.vy

        if neighbors:
            neighbors_vx_avg = neighbors_vx / neighbors
            neighbors_vy_avg = neighbors_vy / neighbors
            self.vx += neighbors_vx_avg * ALIGNMENT_WEIGHT
            self.vy += neighbors_vy_avg * ALIGNMENT_WEIGHT

    def cohesion(self, boids: List["Boid"]) -> None:
        neighbors = 0
        neighbors_dx = 0
        neighbors_dy = 0

        for other in boids:
            if other.id == self.id:
                continue

            dx = other.x - self.x
            dy = other.y - self.y
            distance = math.hypot(dx, dy)
            if PROTECTED_RANGE <= distance <= VISUAL_RANGE:
                neighbors += 1
                neighbors_dx += dx / distance
                neighbors_dy += dy / distance

        if neighbors:
            neighbors_dx_avg = neighbors_dx / neighbors
            neighbors_dy_avg = neighbors_dy / neighbors
            self.vx += neighbors_dx_avg * COHESION_WEIGHT
            self.vy += neighbors_dy_avg * COHESION_WEIGHT

    def draw(self, surface: pygame.Surface) -> None:
        if self.selected:
            center = [self.x, self.y]
            visual_color = pygame.Color(200, 200, 200, a=20)
            protected_color = pygame.Color(169, 169, 169, a=20)
            pygame.draw.circle(surface, visual_color, center, VISUAL_RANGE)
            pygame.draw.circle(surface, protected_color, center, PROTECTED_RANGE)
        front_point = (
            self.x + self.size * math.cos(self.angle),
            self.y + self.size * math.sin(self.angle),
        )
        left_point = (
            self.x + self.size * math.cos(self.angle + math.pi * 0.8),
            self.y + self.size * math.sin(self.angle + math.pi * 0.8),
        )
        right_point = (
            self.x + self.size * math.cos(self.angle - math.pi * 0.8),
            self.y + self.size * math.sin(self.angle - math.pi * 0.8),
        )
        pygame.draw.polygon(surface, BOID_COLOR, [front_point, left_point, right_point])


class Flock:
    def __init__(self, n: int = 150) -> None:
        self.boids = [
            Boid(random.randint(0, SCREEN_SIZE), random.randint(0, SCREEN_SIZE))
            for _ in range(n)
        ]

    def update(self, surface: pygame.Surface) -> None:
        for boid in self.boids:
            boid.boundary_perception()
            boid.separation(self.boids)
            boid.alignment(self.boids)
            boid.cohesion(self.boids)
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
            if event.key == pygame.K_n:
                flock.update(screen)
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for boid in flock.boids:
                distance = math.hypot(mouse_x - boid.x, mouse_y - boid.y)
                if distance < PROTECTED_RANGE:
                    boid.selected = not boid.selected
                    break

    screen.fill(pygame.Color(142, 220, 240, 50))
    if not paused:
        flock.update(screen)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
