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

#define SCREEN_SIZE_X 1920
#define SCREEN_SIZE_Y 1080
#define FPS 30
#define PARTICLE_COUNT 50000
#define PARTICLE_SIZE 2
#define INIT_STATIC true
#define SPEED_LIMIT 5
#define VELOCITY_FACTOR 1
#define FRICTION_FACTOR 0.85
#define ATTRACTION_RADIUS 500
#define REPULSION_RADIUS 15
float ATTRACTION_FACTOR = 20;
float REPULSION_FACTOR = 1;
bool SHOW_INFO = true;
bool DRAW_VECTORS = true;
bool COLOR_MODE_TYPE = true;  // true for "type", false for "velocity"
float PAN_SPEED = 20.0f;

sf::Color getColorByType(int type) {
    switch(type) {
        case 0: return sf::Color(255, 0, 0); // RED
        case 1: return sf::Color(255, 127, 0); // ORANGE
        case 2: return sf::Color(255, 255, 0); // YELLOW
        case 3: return sf::Color(0, 255, 0); // GREEN
        case 4: return sf::Color(0, 0, 255); //BLUE
        case 5: return sf::Color(75, 0, 130); // INDIGO
        default: return sf::Color(148, 0, 211); // VIOLET
    }
}
__constant__ float TYPE_INTERACTIONS[7][7] = {
    {1, -1, 0, 0, 0.2, 0, 0},
    {1, 0.5, 0, 0, -0.2, 0, 0},
    {0, 0.1, 1, -0.3, 0, 0, 0.5},
    {0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2},
    {0.2, -0.2, 0, 0, 1, 0, 0},
    {-0.1, 0, 0, 0, 0, -1, 2},
    {-0.1, 0, -0.2, 0, 0, -0.2, 1},
};
// __constant__ float TYPE_INTERACTIONS[7][7] = {
//     {1, 0.7, -1, 0, -0.1, 0, -0.4},
//     {-0.2, 0.2, 1, 0, 0.5, 0, 0},
//     {0, -0.2, 0.5, 1, 0, 0, 0.8},
//     {0, 0, -0.2, 0.2, 1, 0, 0},
//     {-0.4, 0, 0, -0.2, -1, 1, 0},
//     {0, -1, 0, 0, -0.2, 0.1, 1},
//     {1, 0, -0.8, 0, 0, -0.2, 0.1},
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
    sf::Color shape_color = color;

    if (!COLOR_MODE_TYPE) {
        float speedNormalized = speed / SPEED_LIMIT;
        int speedColor = std::min(255, static_cast<int>(255 * speedNormalized));
        shape_color = sf::Color(255, 255 - speedColor, 255 - speedColor);
    }

    if (DRAW_VECTORS) {
        const float minVectorLength = 5.0f;
        const float maxVectorLength = 20.0f;

        float velocityLength = std::sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1]);
        float normalizedX = velocity[0] / velocityLength;
        float normalizedY = velocity[1] / velocityLength;
        float vectorLength = std::max(minVectorLength, std::min(velocityLength * 4, maxVectorLength));
        float cappedVectorX = normalizedX * vectorLength;
        float cappedVectorY = normalizedY * vectorLength;

        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(position[0], position[1]), shape_color),
            sf::Vertex(sf::Vector2f(position[0] + cappedVectorX, position[1] + cappedVectorY), shape_color)
        };
        window.draw(line, 4, sf::Lines);
    } else {
        sf::CircleShape shape(PARTICLE_SIZE);
        shape.setPosition(position[0], position[1]);
        shape.setFillColor(shape_color);
        window.draw(shape);
    }
}

void Particle::updateFromGPU(float* positions, float* velocities, int idx) {
    position[0] = positions[idx * 2];
    position[1] = positions[idx * 2 + 1];
    velocity[0] = velocities[idx * 2];
    velocity[1] = velocities[idx * 2 + 1];
    speed = std::hypot(velocity[0], velocity[1]);
}

__global__ void updateParticlesKernel(float* positions, float* velocities, int* types, int numParticles, float attraction_factor, float repulsion_factor, int view_width, int view_height) {
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
                forceMagnitude *= repulsion_factor;
            }
            // Attractive force
            else if (REPULSION_RADIUS < distance && distance < ATTRACTION_RADIUS) {
                forceMagnitude = 1 / (distance * distance);
                forceMagnitude *= attraction_factor * TYPE_INTERACTIONS[type][otherType];
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
        if (posX > view_width) {
            posX = view_width;
            velX = -velX;
        }
        if (posY < 0) {
            posY = 0;
            velY = -velY;
        }
        if (posY > view_height) {
            posY = view_height;
            velY = -velY;
        }

        // Update position
        positions[idx * 2] = posX + velX * VELOCITY_FACTOR;
        positions[idx * 2 + 1] = posY + velY * VELOCITY_FACTOR;

        velocities[idx * 2] = velX * FRICTION_FACTOR;
        velocities[idx * 2 + 1] = velY * FRICTION_FACTOR;
    }
}

class System {
public:
    System(int view_width = SCREEN_SIZE_X, int view_height = SCREEN_SIZE_Y, int size = PARTICLE_COUNT, bool initStatic = INIT_STATIC);
    void update(sf::RenderWindow& window, float attraction_factor, float repulsion_factor, int view_width, int view_height);

private:
    std::vector<Particle> particles;
    bool initStatic;

    void createParticle(int view_width, int view_height, std::vector<float> position = std::vector<float>());
    void updateParticlesOnGPU(float attraction_factor, float repulsion_factor, int view_width, int view_height);
};

System::System(int view_width, int view_height, int size, bool initStatic) : initStatic(initStatic) {
    for (int i = 0; i < size; ++i) {
        createParticle(view_width, view_height);
    }
}

void System::createParticle(int view_width, int view_height, std::vector<float> position) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disPosX(0, view_width);
    std::uniform_real_distribution<> disPosY(0, view_height);
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

void System::updateParticlesOnGPU(float attraction_factor, float repulsion_factor, int view_width, int view_height) {
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
    updateParticlesKernel<<<numBlocks, blockSize>>>(d_positions, d_velocities, d_types, numParticles, attraction_factor, repulsion_factor, view_width, view_height);

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

void System::update(sf::RenderWindow& window, float attraction_factor, float repulsion_factor, int view_width, int view_height) {
    updateParticlesOnGPU(attraction_factor, repulsion_factor, view_width, view_height);
    for (auto& particle : particles) {
        particle.draw(window);
    }
}

int main() {
    System system;
    float attraction_factor = ATTRACTION_FACTOR;
    float repulsion_factor = REPULSION_FACTOR;

    int iterations = 0;
    sf::Clock clock;
    sf::RenderWindow window(sf::VideoMode(SCREEN_SIZE_X, SCREEN_SIZE_Y), "Particle System");
    sf::View view = window.getView();
    window.setFramerateLimit(FPS);
    int system_width = SCREEN_SIZE_X;
    int system_height = SCREEN_SIZE_Y;

    // Info text
    sf::Font font;
    font.loadFromFile("../arial.ttf");
    sf::Text paramText;
    paramText.setFont(font);
    paramText.setPosition(10, 10);
    paramText.setCharacterSize(15);
    paramText.setFillColor(sf::Color::White);

    while (window.isOpen()) {
        sf::Vector2f viewSize = view.getSize();
        system_width = max(system_width, static_cast<int>(viewSize.x));
        system_height = max(system_height, static_cast<int>(viewSize.y));
    
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    system_width = static_cast<int>(viewSize.x);
                    system_height = static_cast<int>(viewSize.y);
                    system = System(system_width, system_height, PARTICLE_COUNT, INIT_STATIC);
                    attraction_factor = ATTRACTION_FACTOR;
                    repulsion_factor = REPULSION_FACTOR;
                } else if (event.key.code == sf::Keyboard::I) {
                    SHOW_INFO = !SHOW_INFO;
                } else if (event.key.code == sf::Keyboard::C) {
                    COLOR_MODE_TYPE = !COLOR_MODE_TYPE;
                } else if (event.key.code == sf::Keyboard::V) {
                    DRAW_VECTORS = !DRAW_VECTORS;
                } else if (event.key.code == sf::Keyboard::W) {
                    view.move(0, -PAN_SPEED);
                } else if (event.key.code == sf::Keyboard::S) {
                    view.move(0, PAN_SPEED);
                } else if (event.key.code == sf::Keyboard::A) {
                    view.move(-PAN_SPEED, 0);
                } else if (event.key.code == sf::Keyboard::D) {
                    view.move(PAN_SPEED, 0);
                } else if (event.key.code == sf::Keyboard::E) {
                    view.zoom(0.95);
                    // viewSize = view.getSize();
                    // view.setCenter(viewSize.x/2, viewSize.y/2);
                } else if (event.key.code == sf::Keyboard::Q) {
                    view.zoom(1.05);
                    viewSize = view.getSize();
                    view.setCenter(viewSize.x/2, viewSize.y/2);
                } else if (event.key.code == sf::Keyboard::Up) {
                    attraction_factor += 1;
                } else if (event.key.code == sf::Keyboard::Down) {
                    attraction_factor -= 1;
                } else if (event.key.code == sf::Keyboard::Left) {
                    repulsion_factor -= 0.1;
                } else if (event.key.code == sf::Keyboard::Right) {
                    repulsion_factor += 0.1;
                }
            }
        }

        window.clear(sf::Color::Black);

        // Run simulation
        system.update(window, attraction_factor, repulsion_factor, system_width, system_height);

        // Display information
        if (SHOW_INFO) {
            window.setView(window.getDefaultView());
            std::ostringstream oss;
            sf::Time elapsed = clock.restart();
            float fps = 1.0f / elapsed.asSeconds();
            oss << "Particle Count: " << PARTICLE_COUNT << "\n";
            oss << "Velocity Factor: " << VELOCITY_FACTOR << "\n";
            oss << "Friction Factor: " << FRICTION_FACTOR << "\n";
            oss << "Attraction Factor: " << attraction_factor << "\n";
            oss << "Repulsion Factor: " << repulsion_factor << "\n";
            oss << "Iterations: " << iterations << "\n";
            oss << "FPS: " << std::fixed << std::setprecision(0) << fps << "\n";
            paramText.setString(oss.str());
            window.draw(paramText);
        }

        window.setView(view);
        window.display();
        iterations++;
    }

    return 0;
}
