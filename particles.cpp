#include <SFML/Graphics.hpp>
#include <cmath>
#include <random>
#include <vector>
#include <unordered_map>
#include <tuple>

#define SCREEN_SIZE_X 1280
#define SCREEN_SIZE_Y 720
#define FPS 30
#define PARTICLE_COUNT 1200
#define PARTICLE_SIZE 1
#define VELOCITY_SCALER 2
#define SPEED_LIMIT 6
#define INTERACTION_RADIUS 100
#define REPULSION_RADIUS 15
#define REPULSION_FACTOR 1
#define ATTRACTION_FACTOR 4
#define FRICTION_COEFFICIENT 0.7

#define INIT_STATIC true
bool DRAW_VECTORS = false;
std::string COLOR_MODE = "type";  // "type" or "velocity"

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

const float TYPE_INTERACTIONS[7][7] = {
    {4, -1, 0, 0, 0, 0, 0},
    {1, 4, -1, 0, 0, 0, 0},
    {0, 1, 4, -1, 0, 0, 0},
    {0, 0, 1, 4, -1, 0, 0},
    {0, 0, 0, 1, 4, -1, 0},
    {0, 0, 0, 0, 1, 4, -1},
    {0, 0, 0, 0, 0, 1, 4},
};


struct Particle;

// Custom hash function for std::pair<int, int>
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
    void move(SpatialGrid& grid);
    void draw(sf::RenderWindow& window);

private:
    void boundaryDetection();
    void neighborInteractions(SpatialGrid& grid);
    void capVelocity();
    void updatePosition();
    float getSpeed();
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
    speed = getSpeed();
}

void Particle::move(SpatialGrid& grid) {
    boundaryDetection();
    neighborInteractions(grid);
    capVelocity();
    updatePosition();
}

void Particle::draw(sf::RenderWindow& window) {
    sf::CircleShape shape(PARTICLE_SIZE);
    shape.setPosition(position[0], position[1]);
    if (COLOR_MODE == "velocity") {
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
            sf::Vertex(sf::Vector2f(position[0] + velocity[0] * 8, position[1] + velocity[1] * 8), sf::Color::Red)
        };
        window.draw(line, 2, sf::Lines);
    }
}

void Particle::boundaryDetection() {
    if (position[0] < 0) {
        position[0] = 0;
        velocity[0] = -velocity[0];
    }
    if (position[0] > SCREEN_SIZE_X) {
        position[0] = SCREEN_SIZE_X;
        velocity[0] = -velocity[0];
    }
    if (position[1] < 0) {
        position[1] = 0;
        velocity[1] = -velocity[1];
    }
    if (position[1] > SCREEN_SIZE_Y) {
        position[1] = SCREEN_SIZE_Y;
        velocity[1] = -velocity[1];
    }
}

void Particle::neighborInteractions(SpatialGrid& grid) {
    std::vector<Particle*> neighbors = grid.getNeighboringParticles(this);
    for (Particle* other : neighbors) {
        if (other == this) continue;

        float dx = other->position[0] - position[0];
        float dy = other->position[1] - position[1];
        float distance = std::hypot(dx, dy);
        float angle = std::atan2(dy, dx);
        float forceMagnitude = 0;

        if (distance <= REPULSION_RADIUS) {
            forceMagnitude = distance / REPULSION_RADIUS - 1;
            forceMagnitude *= REPULSION_FACTOR;
        } else if (distance < INTERACTION_RADIUS) {
            float attractionFactor = TYPE_INTERACTIONS[type][other->type];
            forceMagnitude = (distance / INTERACTION_RADIUS) * ATTRACTION_FACTOR - 1;
            forceMagnitude *= attractionFactor;
        }

        velocity[0] += forceMagnitude * std::cos(angle);
        velocity[1] += forceMagnitude * std::sin(angle);
    }
}

void Particle::capVelocity() {
    velocity[0] *= FRICTION_COEFFICIENT;
    velocity[1] *= FRICTION_COEFFICIENT;
    speed = getSpeed();
    if (speed > SPEED_LIMIT) {
        float scalingFactor = SPEED_LIMIT / speed;
        velocity[0] *= scalingFactor;
        velocity[1] *= scalingFactor;
    }
}

void Particle::updatePosition() {
    position[0] += velocity[0] * VELOCITY_SCALER;
    position[1] += velocity[1] * VELOCITY_SCALER;
    speed = getSpeed();
}

float Particle::getSpeed() {
    return std::hypot(velocity[0], velocity[1]);
}

class System {
public:
    System(int size = PARTICLE_COUNT, bool initStatic = INIT_STATIC);
    void update(sf::RenderWindow& window);

private:
    std::vector<Particle> particles;
    bool initStatic;
    SpatialGrid grid;

    void createParticle(std::vector<float> position = std::vector<float>());
};

System::System(int size, bool initStatic) : initStatic(initStatic), grid(INTERACTION_RADIUS) {
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
    grid.addParticle(&particles.back());
}

void System::update(sf::RenderWindow& window) {
    grid = SpatialGrid(INTERACTION_RADIUS);
    for (Particle& particle : particles) {
        grid.addParticle(&particle);
    }
    for (auto& particle : particles) {
        particle.move(grid);
        particle.draw(window);
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(SCREEN_SIZE_X, SCREEN_SIZE_Y), "Particle System");
    window.setFramerateLimit(FPS);

    System system;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    system = System(PARTICLE_COUNT, INIT_STATIC);
                } else if (event.key.code == sf::Keyboard::C) {
                    COLOR_MODE = (COLOR_MODE.compare("type") == 0) ? "velocity" : "type";
                } else if (event.key.code == sf::Keyboard::V) {
                    DRAW_VECTORS = DRAW_VECTORS ? false : true;
                }
            }
        }

        window.clear(sf::Color::Black);
        system.update(window);
        window.display();
    }

    return 0;
}