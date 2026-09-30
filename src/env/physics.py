import pymunk
import random
import math

TEST_BLUEPRINT = {
    "num_of_rows": 4,
    "parking_spots_per_row": 10,
    "car_start": (325, 200, 180),
    "parking_density": 0.8,
    "car_target_spot": (0, 2),
    "car_mass": 100
}

class ParkingPhysics:
    def __init__(self, blueprint):
        self.SPOT_WIDTH = 50
        self.SPOT_DEPTH = 100
        self.VERTICAL_LANE = 150
        self.HORIZONTAL_LANE = 100

        self.PARKED_CAR_LENGTH = self.SPOT_DEPTH - 25
        self.PARKED_CAR_WIDTH = self.SPOT_WIDTH - 15

        self.PLAYER_CAR_LENGTH = self.SPOT_DEPTH - 40
        self.PLAYER_CAR_WIDTH = self.SPOT_WIDTH - 20

        self.blueprint = blueprint

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.1

        self.width, self.height = self.calculate_map_size(blueprint)

        self.borders = [
            [(0, 0), (self.width, 0)],
            [(0, 0), (0, self.height)],
            [(self.width, self.height), (self.width, 0)],
            [(0, self.height), (self.width, self.height)],
        ]

        self.num_of_cars = self.get_num_of_cars(blueprint)
        self.all_parking_spots = self.get_all_parking_spots(blueprint)

        self.parked_cars_indices = random.sample(self.all_parking_spots, self.num_of_cars)
        self.parked_cars_coordinates = self.calculate_spot_coordinates(self.parked_cars_indices)
        self.parked_car_bodies = []

        for border in self.borders:
            wall = pymunk.Segment(self.space.static_body, border[0], border[1], 5)

            self.space.add(wall)

        for parked_car_coordinate in self.parked_cars_coordinates:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = parked_car_coordinate
            shape = pymunk.Poly.create_box(body, (self.PARKED_CAR_LENGTH, self.PARKED_CAR_WIDTH))
            self.space.add(body, shape)
            self.parked_car_bodies.append((body, random.randint(1, 3)))

        self.agent_body = self.spawn_agent(blueprint)

    def step(self, action):
        throttle = action[0]
        steer = action[1]

        vx, vy = self.agent_body.velocity
        angle = self.agent_body.angle

        forward_speed = (vx * math.cos(angle)) + (vy * math.sin(angle))

        self.agent_body.torque = steer * forward_speed * 15
        self.agent_body.apply_force_at_local_point((throttle * 100, 0), (0, 0))
        self.space.step(1 / 60.0)

    def calculate_map_size(self, blueprint):
        map_width = (blueprint['num_of_rows'] * self.SPOT_DEPTH) + ((blueprint['num_of_rows'] + 1) * self.VERTICAL_LANE)
        map_height = (blueprint['parking_spots_per_row'] * self.SPOT_WIDTH) + (2 * self.HORIZONTAL_LANE)

        return map_width, map_height

    def get_num_of_cars(self, blueprint):
        return int((blueprint['num_of_rows'] * blueprint['parking_spots_per_row'] - 1) * blueprint['parking_density'])

    def get_all_parking_spots(self, blueprint):
        spot_list = []

        for row in range(blueprint['num_of_rows']):
            for spot in range(blueprint['parking_spots_per_row']):
                if (row, spot) == blueprint['car_target_spot']:
                    continue
                else:
                    spot_list.append((row, spot))

        return spot_list

    def calculate_spot_coordinates(self, cars_indices):
        cars_coordinates = []

        for indices in cars_indices:
            x = ((indices[0] + 1) * self.VERTICAL_LANE) + (indices[0] * self.SPOT_DEPTH) + (self.SPOT_DEPTH / 2)
            y = self.HORIZONTAL_LANE + (indices[1] * self.SPOT_WIDTH) + (self.SPOT_WIDTH / 2)

            cars_coordinates.append((x, y))

        return cars_coordinates

    def spawn_agent(self, blueprint):
        mass = blueprint['car_mass']
        moment = pymunk.moment_for_box(mass, (self.PLAYER_CAR_LENGTH, self.PLAYER_CAR_WIDTH))
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC, mass=mass, moment=moment)
        body.position = (blueprint['car_start'][0], blueprint['car_start'][1])
        body.angle = math.radians(blueprint['car_start'][2])
        shape = pymunk.Poly.create_box(body, (self.PLAYER_CAR_LENGTH, self.PLAYER_CAR_WIDTH))
        self.space.add(body, shape)

        return body