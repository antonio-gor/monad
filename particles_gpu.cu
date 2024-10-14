#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <random>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <iostream>
#include <iomanip>

// #define SCREEN_SIZE_X 1920
// #define SCREEN_SIZE_Y 1080
#define SCREEN_SIZE_X 1280
#define SCREEN_SIZE_Y 720
#define FPS 30
#define PARTICLE_COUNT 5000
#define PARTICLE_SIZE 1
#define VELOCITY_SCALER 1
#define SPEED_LIMIT 5
#define INTERACTION_RADIUS 300
#define REPULSION_RADIUS 15
#define REPULSION_FACTOR 1
#define ATTRACTION_FACTOR 20
#define FRICTION_COEFFICIENT 0.85
#define INIT_STATIC true
bool DRAW_VECTORS = false;
bool COLOR_MODE_TYPE = true;  // true for "type", false for "velocity"

sf::Color getColorByType(int type) {
    switch(type) {
        case 0: return sf::Color::White;
        case 1: return sf::Color::Yellow;
        case 2: return sf::Color::Green;
        case 3: return sf::Color::Cyan;
        case 4: return sf::Color::Blue;
        case 5: return sf::Color::Magenta;
        default: return sf::Color::Red;
    }
}
__constant__ float TYPE_INTERACTIONS[7][7] = {
    {1, -1, 0, 0, 0.2, 0, 0},
    {1, 1, 0, 0, -0.2, 0, 0},
    {0, 0.2, 1.5, 0, 0, 0, 0.5},
    {0, 0, 0, 1, 0, 0, 0},
    {0.2, -0.2, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, -1, 2},
    {0, 0, 0, 0, 0, -0.2, 1},
};
// __constant__ float TYPE_INTERACTIONS[7][7] = {
//     {1, -1, 0.2, 0, 0, -0.2, -0.2},
//     {0.2, 1, -0.2, 0, 0, 0, -0.2},
//     {-0.6, -1, 4, -1, 0, 0, -1},
//     {0, 0, -1, 0.2, 0, 0, -0.1},
//     {0.2, 0, 0, -0.2, 0, 0.2, -0.2},
//     {0, 1, 0.2, 0, 0, 0, -0.1},
//     {-0.2, -0.2, -0.2, 0.4, 0.2, -0.2, 0.6},
// };
// __constant__ float TYPE_INTERACTIONS[7][7] = {
//     {1, 0, 0, 0, 0, 0, 0},
//     {0, 1, 0, 0, 0, 0, 0},
//     {0, 0, 1, 0, 0, 0, 0},
//     {0, 0, 0, 1, 0, 0, 0},
//     {0, 0, 0, 0, 1, 0, 0},
//     {0, 0, 0, 0, 0, 1, 0},
//     {0, 0, 0, 0, 0, 0, 1},
// };

struct Particle;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

class SpatialGrid {
public:
    SpatialGrid(int cellSize) : cellSize(cellSize) {}

    void addParticle(Particle* particle);
    std::vector<Particle*> getNeighboringParticles(Particle* particle);

private:
    int cellSize;
    std::unordered_map<std::pair<int, int>, std::vector<Particle*>, pair_hash> cells;

    std::pair<int, int> getCellCoords(const std::vector<float>& position);
};

struct Particle {
    std::vector<float> position;
    std::vector<float> velocity;
    float speed;
    int type;
    sf::Color color;

    Particle(std::vector<float> position, std::vector<float> velocity, int type);
    void draw(sf::RenderWindow& window);
    void updateFromGPU(float* positions, float* velocities, int idx);
};

void SpatialGrid::addParticle(Particle* particle) {
    std::pair<int, int> cellCoords = getCellCoords(particle->position);
    cells[cellCoords].push_back(particle);
}

std::pair<int, int> SpatialGrid::getCellCoords(const std::vector<float>& position) {
    return std::make_pair(static_cast<int>(position[0] / cellSize), static_cast<int>(position[1] / cellSize));
}

std::vector<Particle*> SpatialGrid::getNeighboringParticles(Particle* particle) {
    std::vector<Particle*> neighboringParticles;
    std::pair<int, int> cellCoords = getCellCoords(particle->position);
    int cellX = cellCoords.first;
    int cellY = cellCoords.second;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            std::pair<int, int> neighborCell = std::make_pair(cellX + dx, cellY + dy);
            if (cells.find(neighborCell) != cells.end()) {
                neighboringParticles.insert(neighboringParticles.end(), cells[neighborCell].begin(), cells[neighborCell].end());
            }
        }
    }
    return neighboringParticles;
}

Particle::Particle(std::vector<float> position, std::vector<float> velocity, int type)
    : position(position), velocity(velocity), type(type), color(getColorByType(type)) {
    speed = std::hypot(velocity[0], velocity[1]);
}

void Particle::draw(sf::RenderWindow& window) {
    sf::CircleShape shape(PARTICLE_SIZE);
    shape.setPosition(position[0], position[1]);
    if (!COLOR_MODE_TYPE) {
        float speedNormalized = speed / SPEED_LIMIT;
        int speedColor = std::min(255, static_cast<int>(255 * speedNormalized));
        shape.setFillColor(sf::Color(255, 255 - speedColor, 255 - speedColor));
    } else {
        shape.setFillColor(color);
    }
    window.draw(shape);

    if (DRAW_VECTORS) {
        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(position[0], position[1]), sf::Color::Red),
            sf::Vertex(sf::Vector2f(position[0] + velocity[0] * 4, position[1] + velocity[1] * 4), sf::Color::Red)
        };
        window.draw(line, 2, sf::Lines);
    }
}

void Particle::updateFromGPU(float* positions, float* velocities, int idx) {
    position[0] = positions[idx * 2];
    position[1] = positions[idx * 2 + 1];
    velocity[0] = velocities[idx * 2];
    velocity[1] = velocities[idx * 2 + 1];
    speed = std::hypot(velocity[0], velocity[1]);
}

__global__ void updateParticlesKernel(float* positions, float* velocities, int* types, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        float posX = positions[idx * 2];
        float posY = positions[idx * 2 + 1];
        float velX = velocities[idx * 2];
        float velY = velocities[idx * 2 + 1];
        int type = types[idx];

        // Interaction with neighboring particles
        for (int j = 0; j < numParticles; ++j) {
            if (j == idx) continue;

            float otherPosX = positions[j * 2];
            float otherPosY = positions[j * 2 + 1];
            int otherType = types[j];

            float dx = otherPosX - posX;
            float dy = otherPosY - posY;
            float distance = sqrtf(dx * dx + dy * dy);
            float angle = atan2f(dy, dx);
            float forceMagnitude = 0;

            // Repulsive force
            if (distance <= REPULSION_RADIUS) {
                forceMagnitude = distance / REPULSION_RADIUS - 1;
                forceMagnitude *= REPULSION_FACTOR;
            }
            // Attractive force
            else if (REPULSION_RADIUS < distance && distance < INTERACTION_RADIUS) {
                forceMagnitude = 1 / (distance * distance);
                forceMagnitude *= ATTRACTION_FACTOR * TYPE_INTERACTIONS[type][otherType];
            }
        
            velX += forceMagnitude * cosf(angle);
            velY += forceMagnitude * sinf(angle);

            float currentSpeed = sqrtf(velX * velX+ velY * velY);
            if (currentSpeed > SPEED_LIMIT) {
                float scalingFactor = SPEED_LIMIT / currentSpeed;
                velX *= scalingFactor;
                velY *= scalingFactor;
            }
        }

        // Boundary detection
        if (posX < 0) {
            posX = 0;
            velX = -velX;
        }
        if (posX > SCREEN_SIZE_X) {
            posX = SCREEN_SIZE_X;
            velX = -velX;
        }
        if (posY < 0) {
            posY = 0;
            velY = -velY;
        }
        if (posY > SCREEN_SIZE_Y) {
            posY = SCREEN_SIZE_Y;
            velY = -velY;
        }

        // Update position
        positions[idx * 2] = posX + velX * VELOCITY_SCALER;
        positions[idx * 2 + 1] = posY + velY * VELOCITY_SCALER;

        velocities[idx * 2] = velX * FRICTION_COEFFICIENT;
        velocities[idx * 2 + 1] = velY * FRICTION_COEFFICIENT;
    }
}

class System {
public:
    System(int size = PARTICLE_COUNT, bool initStatic = INIT_STATIC);
    void update(sf::RenderWindow& window);

private:
    std::vector<Particle> particles;
    bool initStatic;

    void createParticle(std::vector<float> position = std::vector<float>());
    void updateParticlesOnGPU();
};

System::System(int size, bool initStatic) : initStatic(initStatic) {
    for (int i = 0; i < size; ++i) {
        createParticle();
    }
}

void System::createParticle(std::vector<float> position) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disPosX(0, SCREEN_SIZE_X);
    std::uniform_real_distribution<> disPosY(0, SCREEN_SIZE_Y);
    std::uniform_real_distribution<> disVel(-1, 1);

    if (position.empty()) {
        position.clear(); position.push_back(static_cast<float>(disPosX(gen))); position.push_back(static_cast<float>(disPosY(gen)));
    }
    std::vector<float> velocity;
    if (initStatic) {
        velocity.push_back(0);
        velocity.push_back(0);
    } else {
        velocity.push_back(static_cast<float>(disVel(gen)));
        velocity.push_back(static_cast<float>(disVel(gen)));
    }
    int type = std::uniform_int_distribution<>(0, 6)(gen);

    particles.emplace_back(position, velocity, type);
}

void System::updateParticlesOnGPU() {
    int numParticles = particles.size();
    float* d_positions;
    float* d_velocities;
    int* d_types;

    // Allocate device memory
    cudaMalloc(&d_positions, sizeof(float) * 2 * numParticles);
    cudaMalloc(&d_velocities, sizeof(float) * 2 * numParticles);
    cudaMalloc(&d_types, sizeof(int) * numParticles);

    // Copy data from host to device
    std::vector<float> positions, velocities;
    std::vector<int> types;
    for (auto& particle : particles) {
        positions.push_back(particle.position[0]);
        positions.push_back(particle.position[1]);
        velocities.push_back(particle.velocity[0]);
        velocities.push_back(particle.velocity[1]);
        types.push_back(particle.type);
    }
    cudaMemcpy(d_positions, positions.data(), sizeof(float) * 2 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities.data(), sizeof(float) * 2 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_types, types.data(), sizeof(int) * numParticles, cudaMemcpyHostToDevice);

    // Launch kernel to update particles
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    updateParticlesKernel<<<numBlocks, blockSize>>>(d_positions, d_velocities, d_types, numParticles);

    // Copy data back from device to host
    cudaMemcpy(positions.data(), d_positions, sizeof(float) * 2 * numParticles, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities.data(), d_velocities, sizeof(float) * 2 * numParticles, cudaMemcpyDeviceToHost);

    // Update particles on the host
    for (int i = 0; i < numParticles; i++) {
        particles[i].updateFromGPU(positions.data(), velocities.data(), i);
    }

    // Free device memory
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_types);
}

void System::update(sf::RenderWindow& window) {
    updateParticlesOnGPU();
    for (auto& particle : particles) {
        particle.draw(window);
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(SCREEN_SIZE_X, SCREEN_SIZE_Y), "Particle System");
    window.setFramerateLimit(FPS);

    System system;
    sf::Clock clock;

    sf::Font font;
    font.loadFromFile("../arial.ttf");
    sf::Text fps_text;
    fps_text.setFont(font);
    fps_text.setCharacterSize(10); // in pixels, not points
    // fps_text.setStyle(sf::Text::Bold);
    fps_text.setFillColor(sf::Color::White);
    fps_text.setPosition(5, 5);
    std::string fps_value;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    system = System(PARTICLE_COUNT, INIT_STATIC);
                } else if (event.key.code == sf::Keyboard::C) {
                    COLOR_MODE_TYPE = !COLOR_MODE_TYPE;
                } else if (event.key.code == sf::Keyboard::V) {
                    DRAW_VECTORS = !DRAW_VECTORS;
                }
            }
        }

        window.clear(sf::Color::Black);

        // Run simulation
        system.update(window);

        // Log FPS in command line
        sf::Time elapsed = clock.restart();
        float fps = 1.0f / elapsed.asSeconds();
        std::cout << "fps: " << std::fixed << std::setprecision(2) << fps << "\r";
        std::cout.flush();

        // Display FPS in window
        fps_value  = std::to_string(fps);
        fps_text.setString(fps_value);
        window.draw(fps_text);
        window.display();
    }

    return 0;
}
