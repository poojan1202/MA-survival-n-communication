from typing import (
    Type, TypeVar, Sequence, Optional, Tuple, List, Callable, Any, Dict,
    Union, NamedTuple)

from Box2D import (  # type: ignore
    b2World, b2ContactListener, b2Body, b2Fixture, b2FixtureDef, b2Shape,
    b2CircleShape, b2PolygonShape, b2Joint, b2RayCastCallback, b2Vec2,
    b2Mat22, b2Transform, b2_staticBody, b2_dynamicBody, b2AABB,
    b2QueryCallback, b2ChainShape, b2EdgeShape)

_T = TypeVar('_T')

# basic geometry

Vec2 = Union[Tuple[float, float], b2Vec2]
Vec3 = Tuple[float, float, float]


def from_polar(length: float, angle: float) -> b2Vec2:
    R = b2Mat22()
    R.angle = angle
    return R * b2Vec2(length, 0.)


def transform(translation: b2Vec2 = b2Vec2(0, 0), angle: float = 0):
    T = b2Transform()
    T.Set(position=translation, angle=angle)
    return T


def copy_shape(shape: b2Shape):
    if isinstance(shape, b2CircleShape):
        return _copy_circle_shape(shape)
    if isinstance(shape, b2PolygonShape):
        return _copy_polygon_shape(shape)
    if isinstance(shape, b2ChainShape):
        return _copy_chain_shape(shape)
    if isinstance(shape, b2EdgeShape):
        return _copy_edge_shape(shape)


def _copy_circle_shape(shape: b2CircleShape):
    return b2CircleShape(radius=shape.radius)


def _copy_polygon_shape(shape: b2PolygonShape):
    newshape = b2PolygonShape(vertices=shape.vertices)
    return newshape


def _copy_chain_shape(shape: b2ChainShape):
    return b2ChainShape(vertices=shape.vertices)


def _copy_edge_shape(shape: b2EdgeShape):
    return b2EdgeShape(vertices=shape.vertices)


def square_shape(side: float) -> b2Shape:
    return b2PolygonShape(box=(side / 2., side / 2.))


def rect_shape(width: float, height: float) -> b2Shape:
    return b2PolygonShape(box=(width / 2., height / 2.))


def circle_shape(radius: float) -> b2Shape:
    return b2CircleShape(radius=radius)


# Return width and height of a shape assuming it's a box.
def rect_dimensions(shape: b2PolygonShape) -> Tuple[float, float]:
    vertices = [b2Vec2(v) for v in shape.vertices]
    width, height = 0, 0
    for v, w in zip(vertices, [*vertices[1:], vertices[0]]):
        if width == 0 and v[0] != w[0] and v[1] == w[1]:
            width = (w - v).length
        if height == 0 and v[1] != w[1] and v[0] == w[0]:
            height = (w - v).length
    return width, height


# creating bodies with modular behaviour

# A module is associated to a group of bodies, which in turn may have several 
# modules associated to it; each module may control the behaviour of bodies in 
# the group by defining one or more methods of the Module class.
# Each body always knows which group it belongs to; by accessing the group, 
# one can spawn and despawn bodies in that group and access their module-
# specific data (if any), so Box2D methods should not be used for that since 
# they mess up the group state.
# A more advanced case is when a module is used in more than one group; in 
# that case, it is its responsibility to distinguish between groups when its 
# methods are called.
# Things NOT to do inside module methods (these WILL mess up the group state):
# - change the list of bodies, modules or set the world attribute of the group
# - create or destroy a body that belongs in a group without using the group's 
#   spawn or despawn methods
# - change the userData of a body that belongs in a group
# Things that can be done safely:
# - perform raycasts on group.world
# - [...]
# Good examples of usage are in the semantics module.
class Module:
    # called when sim is reset
    def post_reset(self, group: 'Group'):
        pass

    # called before each sim step
    def pre_step(self, group: 'Group'):
        pass

    # called after each sim step
    def post_step(self, group: 'Group'):
        pass

    # called when new bodies spawn in the group (they wont yet be in the
    # group.bodies)
    def post_spawn(self, bodies: List[b2Body]):
        pass

    # called when some bodies are about to despawn from the group
    def pre_despawn(self, bodies: List[b2Body]):
        pass


# utiltiy type to pack Box2D body definitions

default_density = 1
default_restitution = 0
default_damping = 0.8


class Prototype(NamedTuple):
    shape: b2Shape
    dynamic: bool = True
    density: float = default_density
    restitution: float = default_restitution
    damping: float = default_damping
    sensor: bool = False


def prototype(body: b2Body) -> Prototype:
    return Prototype(
        shape=copy_shape(body.fixtures[0].shape),
        dynamic=body.type is b2_dynamicBody,
        density=body.fixtures[0].density,
        restitution=body.fixtures[0].restitution,
        damping=body.linearDamping,
        sensor=body.fixtures[0].sensor)


# either position or position and orientation
Placement = Union[Vec3, Vec2]


class Group:
    bodies: List[b2Body]
    modules: List[Module]
    world: b2World

    @staticmethod
    def body_group(body: b2Body) -> 'Group':
        return body.userData

    # this is less efficient than self.despawn, so use that if possible
    @staticmethod
    def despawn_body(body: b2Body):
        Group.body_group(body).despawn([body])

    def __init__(self, modules: Sequence[Module] = []):
        self.bodies = []
        self.modules = []
        self.modules += modules

    def get(self, type_: Type[_T]) -> List[_T]:
        return [m for m in self.modules if isinstance(m, type_)]

    def reset(self, world: b2World):
        # TODO destroy all bodies in the previous world?
        self.world = world
        self.bodies = []
        for module in self.modules:
            module.post_reset(self)

    def pre_step(self):
        for module in self.modules:
            module.pre_step(self)

    def post_step(self):
        for module in self.modules:
            module.post_step(self)

    def spawn(self, prototypes: List[Prototype], placements: List[Placement]):
        bodies = [self._create_body(proto, pos)
                  for proto, pos in zip(prototypes, placements)]
        for module in self.modules:
            module.post_spawn(bodies)
        self.bodies += bodies

    def despawn(self, bodies: List[b2Body]):
        for module in self.modules:
            module.pre_despawn(bodies)
        self.bodies = [b for b in self.bodies if b not in bodies]
        [self._destroy_body(b) for b in bodies]

    def _create_body(self, prototype: Prototype, placement: Placement):
        has_angle = len(placement) == 3
        position = placement if not has_angle else placement[0:2]
        angle = 0 if not has_angle else placement[2]  # type: ignore
        type = b2_dynamicBody if prototype.dynamic else b2_staticBody
        fixture = b2FixtureDef(
            shape=prototype.shape, density=prototype.density,
            restitution=prototype.restitution, isSensor=prototype.sensor)
        body = self.world.CreateBody(
            type=type, position=position, angle=angle, fixtures=fixture,
            linearDamping=prototype.damping, angularDamping=prototype.damping,
            userData=self)
        return body

    def _destroy_body(self, body: b2Body):
        self.world.DestroyBody(body)


class Simulation:
    world: b2World
    substeps: int
    time_step: float
    velocity_iterations: int
    position_iterations: int
    groups: Dict[str, Group]

    def __init__(
            self, substeps: int = 2, time_step: float = 1 / 60,
            velocity_iterations: int = 10, position_iterations: int = 10,
            groups: Dict[str, Group] = {}):
        self.substeps = substeps
        self.time_step = time_step
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self.groups = groups

    def reset(self):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        for group in self.groups.values():
            group.reset(self.world)

    def step(self):
        for group in self.groups.values():
            group.pre_step()
        for _ in range(self.substeps):
            self.world.Step(
                self.time_step, self.velocity_iterations,
                self.position_iterations)
        self.world.ClearForces()
        for group in self.groups.values():
            group.post_step()


# basic concrete modules

# utility module to assign indices to bodies in a group which are consistent 
# also when bodies despawn; indices are assigned on reset based on the bodies 
# currently in the group, so this module should come after modules that spawn 
# any body that needs an index. This also conveniently tracks which bodies 
# died from the last reset. The post_reset method can be also called directly 
# to reset the indices using the bodies currently in the given group. NOTE 
# that to get the index of a body that is about to die, a module must come 
# before this since None will be assigned to its spot after this module's 
# pre_despawn.
class IndexBodies(Module):
    bodies: List[Optional[b2Body]]

    def post_reset(self, group: Group):
        self.bodies = list(group.bodies)

    def pre_despawn(self, bodies: List[b2Body]):
        for body in bodies:
            # The if is needed to avoid double-despawn errors.
            # TODO optimize this by eliminating double-despawns (e.g. 2 agents pickup the same item in the same time step)
            if body in self.bodies:
                self.bodies[self.bodies.index(body)] = None


# adds deaths to a buffer (not with body objects, but with body indices) that can be flushed with flush method
class TrackDeaths(Module):

    def __init__(self, index_module: IndexBodies = None):
        self.index_module = index_module

    def post_reset(self, group: Group):
        if self.index_module is None:
            self.index_module = group.get(IndexBodies)[0]
        self.deaths = []

    def pre_despawn(self, bodies: b2Body):
        for body in bodies:
            dead_id = self.index_module.bodies.index(body)
            self.deaths.append(dead_id)

    def flush(self):
        deaths = self.deaths
        self.deaths = []
        return deaths


# prints a death message when bodies despawn from the group, using body
# indices if available; mostly useful for debugging
class LogDeaths(Module):

    def post_reset(self, group: Group):
        self.group = group

    def pre_despawn(self, bodies: b2Body):
        index_bodies = self.group.get(IndexBodies)
        if len(index_bodies) == 0:
            for body in bodies:
                print(f'A body {body} despawned.')
            return
        for m in index_bodies:
            for body in bodies:
                index = m.bodies.index(body)
                print(f'Body #{index} despawned.')


# 2D "cameras", i.e. detects bodies with LOS inside its FOV, assuming the
# cameras are placed at the positions of bodies in the group. For the sake of 
# simplicity, 2 approximations are used: LOS is only checked to for the 
# position of each body (not the full shape), and the vision cone is actually 
# a triangle (instead of a slice of a circle).
class Cameras(Module):
    depth: float
    fov: float
    vision_cone: b2PolygonShape
    seen: List[List[b2Body]]

    def __init__(self, depth: float, fov: float):
        self.depth = depth
        self.fov = fov
        left = from_polar(self.depth, +self.fov / 2)
        center = b2Vec2(self.depth, 0)
        right = from_polar(self.depth, -self.fov / 2)
        self.vision_cone = b2PolygonShape(
            vertices=[b2Vec2(0, 0), left, center, right])

    def post_reset(self, group: Group):
        self._update_seen(group)

    def post_step(self, group: Group):
        self._update_seen(group)

    def _update_seen(self, group: Group):
        self.seen = []
        for body in group.bodies:
            # Get all bodies inside the vision cone.
            others = shape_query(
                group.world, self.vision_cone, body.transform)
            # Check LOS.
            self.seen.append([])
            for other in others:
                if other == body:
                    continue
                d = other.position - body.position
                # Add a small epsilon to ensure the laser scan hits.
                end = body.position + (1 + 1e-6) * d
                scan = laser_scan(group.world, body.position, end)
                if scan is None:
                    pass
                elif scan[0].body == other:  # type: ignore
                    self.seen[-1].append(scan[0].body)  # type: ignore


class Lidars(Module):
    n_lasers: int
    fov: float
    depth: float
    scans: List[List['LaserScan']]
    origins: List[b2Vec2]
    endpoints: List[List[b2Vec2]]

    def __init__(self, n_lasers: int, fov: float, depth: float):
        self.n_lasers = n_lasers
        self.fov = fov
        self.depth = depth

    def post_reset(self, group: Group):
        self._update(group)

    def post_step(self, group: Group):
        self._update(group)

    def _update(self, group: Group):
        self.origins = [body.position for body in group.bodies]
        orientations = [body.angle for body in group.bodies]
        self.endpoints = [self._endpoints(p, a)
                          for p, a in zip(self.origins, orientations)]
        self.scans = [[laser_scan(group.world, a, b) for b in bs]
                      for a, bs in zip(self.origins, self.endpoints)]

    def _endpoints(self, origin: b2Vec2, orientation: float):
        endpoints = []
        for i in range(self.n_lasers):
            angle = i * (self.fov / (self.n_lasers - 1)) - self.fov / 2.
            angle += orientation
            endpoint = origin + from_polar(length=self.depth, angle=angle)
            endpoints.append(endpoint)
        return endpoints


class DynamicMotors(Module):
    # (*linear,angular) impulses given for unitary controls
    impulse: Vec3
    # whether to avoid resetting the controls to 0 after each application
    drift: bool
    controls: List[Vec3] = []

    def __init__(self, impulse: Vec3, drift: bool = False):
        self.impulse = impulse
        self.drift = drift

    # completes missing controls with 0s, ignores excess controls
    def pre_step(self, group: Group):
        missing = len(group.bodies) - len(self.controls)
        if missing > 0:
            self.controls += [(0, 0, 0)] * missing
        for body, control in zip(group.bodies, self.controls):
            self._apply_control(body, control)
        if not self.drift:
            self.controls = []

    def _apply_control(self, body: b2Body, control: Vec3):
        R = body.transform.R
        p = body.worldCenter
        parallel_impulse = control[0] * self.impulse[0]
        normal_impulse = control[1] * self.impulse[1]
        linear_impulse = R * b2Vec2(parallel_impulse, normal_impulse)
        angular_impulse = control[2] * self.impulse[2]
        body.ApplyLinearImpulse(linear_impulse, p, True)
        body.ApplyAngularImpulse(angular_impulse, True)


# utilities

LaserScan = Optional[Tuple[b2Fixture, float]]


def laser_scan(world: b2World, start: Vec2, end: Vec2) -> LaserScan:
    start, end = b2Vec2(start), b2Vec2(end)
    raycast = LaserRayCastCallback()
    world.RayCast(raycast, start, end)
    if raycast.relative_depth is None:
        return None
    depth = raycast.relative_depth
    assert (raycast.fixture is not None)
    return raycast.fixture, depth


# returns all bodies whose center overalps with the given shape; it should be
# more efficient than iterating over all bodies and shape testing
# only works with non-compound convex shapes
def shape_query(
        world: b2World, shape: b2Shape,
        transform: b2Transform) -> List[b2Body]:
    # Only test bodies that are in the shape's AABB to avoid checking all 
    # bodies in the world.
    aabb = shape.getAABB(transform, 0)
    callback = AllQueryCallback()
    world.QueryAABB(callback, aabb)
    bodies = []
    for fixture in callback.fixtures:
        body = fixture.body
        if shape.TestPoint(transform, body.worldCenter):
            bodies.append(body)
    return bodies


def aabb_query(
        world: b2World, center: b2Vec2, width: float,
        height: float) -> List[b2Fixture]:
    lo = b2Vec2(center[0] - width / 2, center[1] - height)
    hi = b2Vec2(center[0] + width / 2, center[1] + height)
    aabb = b2AABB(lowerBound=lo, upperBound=hi)
    callback = AllQueryCallback()
    world.QueryAABB(callback, aabb)
    return callback.fixtures


# utiltiy class from https://github.com/pybox2d/pybox2d/wiki/manual#ray-casts
# only retains the closest fixture found by the ray cast
class LaserRayCastCallback(b2RayCastCallback):
    fixture: Optional[b2Fixture] = None
    point: Optional[b2Vec2] = None
    normal: Optional[b2Vec2] = None
    relative_depth: Optional[float] = None

    def __init__(self) -> None:
        b2RayCastCallback.__init__(self)

    # TODO type hints
    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        self.relative_depth = fraction
        return fraction


# detects all fixtures in an AABB region (inspired from https://github.com/pybox2d/pybox2d/wiki/manual#exploring-the-world)
class AllQueryCallback(b2QueryCallback):
    fixtures: List[b2Fixture]

    def __init__(self):
        self.fixtures = []
        b2QueryCallback.__init__(self)

    def ReportFixture(self, fixture):
        self.fixtures.append(fixture)
        return True
