"""https://dl.acm.org/doi/pdf/10.1145/37401.37406"""

import pygame
import random
import math

FPS = 60
SCREEN_SIZE = 800
BOID_SIZE = 8
BOID_COLOR = "brown"
BOID_SPEED = 1.5

# pygame setup
pygame.init()
pygame.display.set_caption("Boids")
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
running = True
paused = False
next_step = False
dt = 0


class Boid:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = BOID_SPEED
        self.size = BOID_SIZE + random.uniform(-BOID_SIZE / 4, BOID_SIZE / 4)

    def update(self) -> None:
        if self.x < 0 or self.x > SCREEN_SIZE:
            self.direction = math.pi - self.direction
        if self.y < 0 or self.y > SCREEN_SIZE:
            self.direction = -self.direction

        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed

    def draw(self, surface: pygame.Surface) -> None:
        front_point = (
            self.x + self.size * math.cos(self.direction),
            self.y + self.size * math.sin(self.direction),
        )
        left_point = (
            self.x + self.size * math.cos(self.direction + 0.8 * math.pi),
            self.y + self.size * math.sin(self.direction + 0.8 * math.pi),
        )
        right_point = (
            self.x + self.size * math.cos(self.direction - 0.8 * math.pi),
            self.y + self.size * math.sin(self.direction - 0.8 * math.pi),
        )
        pygame.draw.polygon(surface, BOID_COLOR, [front_point, left_point, right_point])


class Flock:
    def __init__(self, n: int = 15) -> None:
        self.boids = [
            Boid(random.randint(0, SCREEN_SIZE), random.randint(0, SCREEN_SIZE))
            for _ in range(n)
        ]


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
    for boid in flock.boids:
        if not paused:
            boid.update()
        boid.draw(screen)
    pygame.display.flip()

    dt = clock.tick(FPS) / 1000

pygame.quit()
