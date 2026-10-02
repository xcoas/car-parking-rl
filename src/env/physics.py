import pymunk
import random
import math
import numpy as np

class ParkingPhysics:
    def __init__(self, blueprint=None, difficulty: float = 0.0):
        self.SPOT_WIDTH = 50
        self.SPOT_DEPTH = 100
        self.VERTICAL_LANE = 150
        self.HORIZONTAL_LANE = 100

        self.PARKED_CAR_LENGTH = self.SPOT_DEPTH - 25
        self.PARKED_CAR_WIDTH = self.SPOT_WIDTH - 15

        self.PLAYER_CAR_LENGTH = self.SPOT_DEPTH - 40
        self.PLAYER_CAR_WIDTH = self.SPOT_WIDTH - 20

        if blueprint is None:
            self.blueprint = self.generate_blueprint(difficulty)
        else:
            self.blueprint = blueprint

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.1

        self.crashed = False
        self.space.on_collision(1, 2, begin=self.handle_collision)
        self.space.on_collision(1, 3, begin=self.handle_collision)

        self.is_parked = False

        self.sensor_angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]

        self.width, self.height = self.calculate_map_size(self.blueprint)

        self.borders = [
            [(0, 0), (self.width, 0)],
            [(0, 0), (0, self.height)],
            [(self.width, self.height), (self.width, 0)],
            [(0, self.height), (self.width, self.height)],
        ]

        self.num_of_cars = self.get_num_of_cars(self.blueprint)
        self.all_parking_spots = self.get_all_parking_spots(self.blueprint)

        self.parked_cars_indices = random.sample(self.all_parking_spots, self.num_of_cars)
        self.parked_cars_coordinates = self.calculate_spot_coordinates(self.parked_cars_indices)
        self.parked_car_bodies = []

        self.parking_spot_coordinate = self.calculate_spot_coordinates([self.blueprint['car_target_spot']])[0]

        for border in self.borders:
            wall = pymunk.Segment(self.space.static_body, border[0], border[1], 5)
            wall.collision_type = 3

            self.space.add(wall)

        for parked_car_coordinate in self.parked_cars_coordinates:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = parked_car_coordinate
            shape = pymunk.Poly.create_box(body, (self.PARKED_CAR_LENGTH, self.PARKED_CAR_WIDTH))
            shape.collision_type = 2
            self.space.add(body, shape)
            self.parked_car_bodies.append((body, random.randint(1, 3)))

        self.agent_body = self.spawn_agent(self.blueprint)
        self.radar_rays = []

    def step(self, action):
        throttle = action[0]
        steer = action[1]

        vx, vy = self.agent_body.velocity
        angle = self.agent_body.angle

        forward_speed = (vx * math.cos(angle)) + (vy * math.sin(angle))

        self.agent_body.torque = steer * forward_speed * 15
        self.agent_body.apply_force_at_local_point((throttle * 100, 0), (0, 0))

        self.space.step(1 / 60.0)

        dx = abs(self.parking_spot_coordinate[0] - self.agent_body.position.x)
        dy = abs(self.parking_spot_coordinate[1] - self.agent_body.position.y)

        speed = self.agent_body.velocity.length
        is_straight = abs(math.cos(self.agent_body.angle)) >= 0.95

        if (dx <= self.SPOT_DEPTH / 5) and (dy <= self.SPOT_WIDTH / 5) and is_straight and (speed < 5.0):
            self.is_parked = True

        self.get_radar_readings()

    def generate_blueprint(self, difficulty: float = 0.0, num_of_rows: int = 4, spot_per_row: int = 10):
        target_row = random.randint(0, num_of_rows - 1)
        target_spot = random.randint(0, spot_per_row - 1)
        target_indices = (target_row, target_spot)

        target_x, target_y = self.calculate_spot_coordinates([target_indices])[0]
        parking_density = np.clip(round(difficulty, 2), 0.2, 1.0)

        lane_side = random.choice([0, 1])
        if lane_side == 0:
            lane_center_x = target_x - (self.SPOT_DEPTH / 2) - (self.VERTICAL_LANE / 2)
            base_angle = 0
        else:
            lane_center_x = target_x + (self.SPOT_DEPTH / 2) + (self.VERTICAL_LANE / 2)
            base_angle = 180

        if difficulty < 0.2:
            start_x = lane_center_x + random.uniform(-10, 10)
            start_y = target_y + random.uniform(-15, 15)
            start_angle = base_angle + random.uniform(-30, 30)
        elif difficulty < 0.6:
            max_y_offset = 250 * difficulty
            start_x = lane_center_x + random.uniform(-20, 20)
            start_y = target_y + random.uniform(-max_y_offset, max_y_offset)
            start_angle = random.choice([90, 270]) + random.uniform(-45 * difficulty, 45 * difficulty)
        else:
            random_lane = random.randint(0, num_of_rows)
            start_x = (random_lane * (self.VERTICAL_LANE + self.SPOT_DEPTH)) + (self.VERTICAL_LANE / 2) + random.uniform(-25, 25)

            _, map_height = self.calculate_map_size({
                'num_of_rows': num_of_rows,
                'parking_spots_per_row': spot_per_row
            })
            start_y = random.uniform(self.HORIZONTAL_LANE / 2, map_height - (self.HORIZONTAL_LANE / 2))
            start_angle = random.uniform(0, 360)

        _, map_height = self.calculate_map_size({
            'num_of_rows': num_of_rows,
            'parking_spots_per_row': spot_per_row
        })
        start_y = max(50, min(map_height - 50, start_y))

        return {
            "num_of_rows": num_of_rows,
            "parking_spots_per_row": spot_per_row,
            "car_start": (start_x, start_y, start_angle),
            "parking_density": parking_density,
            "car_target_spot": target_indices,
            "car_mass": 100
        }

    def handle_collision(self, arbiter, space, data):
        self.crashed = True
        return True

    def get_radar_readings(self):
        max_distance = 200.0
        radar_readings = []
        angle = self.agent_body.angle
        pos_x, pos_y = self.agent_body.position
        sensor_angles = [math.radians(deg) for deg in self.sensor_angles]

        self.radar_rays = []

        for sensor_angle in sensor_angles:
            total_angle = angle + sensor_angle

            end_x = pos_x + max_distance * math.cos(total_angle)
            end_y = pos_y + max_distance * math.sin(total_angle)

            hit = self.space.segment_query_first(
                (pos_x, pos_y),
                (end_x, end_y),
                1.0,
                pymunk.ShapeFilter(group=1)
            )

            if hit is not None:
                radar_readings.append(hit.alpha)
                self.radar_rays.append(((pos_x, pos_y), (hit.point.x, hit.point.y), True))
            else:
                radar_readings.append(1.0)
                self.radar_rays.append(((pos_x, pos_y), (end_x, end_y), False))

        return radar_readings

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
        shape.collision_type = 1
        shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(body, shape)

        return body