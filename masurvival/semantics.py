import math
from typing import (
    Any, Type, Callable, Union, Optional, Tuple, List, Dict, Set, NamedTuple)
from operator import attrgetter

from Box2D import (  # type: ignore
    b2Vec2, b2World, b2Body, b2Shape, b2CircleShape, b2Transform)

import numpy as np

import masurvival.simulation as sim


# prototype for various bodies

def agent_prototype(agent_size: float) -> sim.Prototype:
    shape = sim.circle_shape(radius=agent_size / 2)
    return sim.Prototype(shape=shape)


def box_prototype(box_size: float) -> sim.Prototype:
    shape = sim.rect_shape(width=box_size, height=box_size)
    return sim.Prototype(shape=shape, dynamic=False)


def item_prototype(item_size: float):
    return sim.Prototype(sim.circle_shape(radius=item_size / 2))


# the battle royale mode: the game ends when there is only 1 agent/team left
# alive, which wins
# TODO implement teams
class BattleRoyale(sim.Module):
    over: bool
    # for each body in the body indices, True if it won False if it lost
    results: List[bool]

    def post_reset(self, group: sim.Group):
        self.over = False

    def post_step(self, group: sim.Group):
        if len(group.bodies) > 1:
            return
        body_indices = group.get(sim.IndexBodies)
        assert (len(body_indices) == 1)
        self.over = True
        self.results = [body is not None for body in body_indices[0].bodies]


# random spawning

# astract interface
class Spawner:
    def reset(self):
        pass

    def placements(self, n: int) -> List[b2Vec2]:
        return [b2Vec2(0, 0)] * n


# expects the user to set rng attr explicitely before calling reset
class SpawnGrid(Spawner):
    grid_size: int
    floor_size: float
    rng: np.random.Generator
    positions: List[b2Vec2]
    occupied: List[b2Vec2]

    def __init__(self, grid_size: int, floor_size: float):
        self.grid_size = grid_size
        self.floor_size = floor_size

    def reset(self):
        self.positions = square_grid(self.grid_size, self.floor_size)
        self.occupied = []
        self.rng.shuffle(self.positions)

    def placements(self, n: int) -> List[b2Vec2]:
        occupied = [self.positions.pop() for _ in range(n)]
        self.occupied += occupied
        return occupied


# spawns N bodies with given prototype on world reset
class ResetSpawns(sim.Module):

    def __init__(
            self, n_spawns: int,
            prototype: Union[sim.Prototype, List[sim.Prototype]],
            spawner: Spawner
    ):
        self.n_spawns = n_spawns
        self.prototypes = [prototype] * self.n_spawns
        self.spawner = spawner

    def post_reset(self, group: sim.Group):
        group.spawn(self.prototypes, self.spawner.placements(self.n_spawns))


# randomizes the shapes in the prototypes of a ResetSpawns as box shapes
class RandomizeBoxShapes(sim.Module):

    def __init__(self, avg_w, std_w, avg_h, std_h, min_w=0.1, min_h=0.1):
        self.avg_w = avg_w
        self.std_w = std_w
        self.avg_h = avg_h
        self.std_h = std_h
        self.min_w = min_w
        self.min_h = min_h

    def post_reset(self, group: sim.Group):
        spawn_modules = group.get(ResetSpawns)
        for spawns in spawn_modules:
            for i, proto in enumerate(spawns.prototypes):
                w = max(
                    self.rng.normal(loc=self.avg_w, scale=self.std_w),
                    self.min_w
                )
                h = max(
                    self.rng.normal(loc=self.avg_h, scale=self.std_h),
                    self.min_h
                )
                spawns.prototypes[i] = \
                    proto._replace(shape=sim.rect_shape(width=w, height=h))


# items and inventories

# marks bodies in its group as items (they are immaterial and can interact
# with inventories).
class Item(sim.Module):
    # can be accessed by subclasses
    group: sim.Group
    # should be set by subclasses
    prototype: sim.Prototype
    # for use in subclasses, stores data specific to items with a body; by
    # default, it stores None for each body
    data: Dict[b2Body, Any]

    # this should be overridden by subclasses
    def use(self, user: b2Body, data: Any):
        pass

    # can be used to "drop" (i.e. spawn) an item of this type with the given
    # placement and data; can also be overridden by subclasses if needed
    def drop(self, placement: sim.Placement, data: Any):
        self.group.spawn([self.prototype], [placement])
        # NOTE: this relies on the fact that no other body can spawn in
        # between the above call to Group.spawn and this line, so that the
        # last spawned body will be the spawned item.
        self.data[self.group.bodies[-1]] = data

    def post_reset(self, group: sim.Group):
        # keep a ref to the group so that items can be dropped
        self.group = group
        self.data = {}

    def post_spawn(self, bodies: List[b2Body]):
        for body in bodies:
            for fixture in body.fixtures:
                fixture.sensor = True
            self.data[body] = None


# adds a list inventory with a fixed number of slots to the bodies in its
# group; no other functionality is added, but the pickup, use, drop and
# drop_all methods can be used by other modules to manipulate inventories
# TODO make it so that a single item can have multiple uses, each with their own data
class Inventory(sim.Module):
    slots: int
    inventories: Dict[b2Body, List[Tuple[Item, Any]]]
    on_use: Optional[Callable]

    def __init__(self, slots: int, on_use: Optional[Callable] = None):
        self.slots = slots
        self.on_use = on_use

    def full(self, body: b2Body):
        return len(self.inventories[body])

    # returns False when inventory did not have enough empty slots
    def take(self, body: b2Body, items: List[Item], data: List) -> bool:
        assert len(items) == len(data)
        inventory = self.inventories[body]
        if len(items) <= 0:
            return False
        if len(items) + len(inventory) > self.slots:
            return False
        inventory += list(zip(items, data))  # type: ignore
        return True

    def pickup(self, body: b2Body, item: b2Body):
        items = sim.Group.body_group(item).get(Item)
        data = [m.data[item] for m in items]
        if self.take(body, items, data):
            sim.Group.despawn_body(item)

    def give(self, src: b2Body, dest: b2Body, slot: int = -1):
        dest_inventories = sim.Group.body_group(dest).get(Inventory)
        if len(dest_inventories) == 0:
            return
        try:
            item, data = self.inventories[src].pop(slot)
        except IndexError:
            return
        for inventory in dest_inventories:
            inventory.take(dest, [item], [data])

    def use(self, user: b2Body, slot: int = -1):
        try:
            item, data = self.inventories[user].pop(slot)
        except IndexError:
            return
        if self.on_use is not None:
            self.on_use(item, user, data)
        item.use(user, data)

    def drop(
            self, body: b2Body, slot: int = -1,
            offset: sim.Vec2 = b2Vec2(0, 0)
    ):
        try:
            item, data = self.inventories[body].pop(slot)
        except IndexError:
            return
        item.drop(body.position + offset, data)

    # more efficient alternative to dropping all items (inventory remains
    # but empty)
    def drop_all(
            self, body: b2Body,
            offsets: Union[sim.Vec2, List[sim.Vec2]] = b2Vec2(0, 0)):
        inventory = self.inventories[body]
        if not isinstance(offsets, list):
            offsets = [offsets] * len(inventory)
        for (item, data), offset in zip(inventory, offsets):
            item.drop(body.position + offset, data)
        self.inventories[body] = []

    def post_spawn(self, bodies: List[b2Body]):
        if not hasattr(self, 'inventories'):
            self.inventories = {}
        for body in bodies:
            self.inventories[body] = []

    def pre_despawn(self, bodies: List[b2Body]):
        for body in bodies:
            del self.inventories[body]


# tracks use stats, divided by item type
class TrackUse(sim.Module):

    def post_reset(self, group: sim.Group):
        for m in group.get(Inventory):
            m.on_use = self
        self.uses = {}

    # on_use callback for inventory
    def __call__(self, item, user, data):
        item_class = type(item)
        if item_class not in self.uses:
            self.uses[item_class] = 0
        self.uses[item_class] += 1

    def flush(self):
        uses = self.uses
        self.uses = {}
        return uses


# makes bodies pickup items in a shape centered on them, putting them in
# the first inventory module it finds in the group
class AutoPickup(sim.Module):
    # the transform of each body will be applied to it
    shape: b2Shape

    def __init__(self, shape: b2Shape):
        self.shape = shape

    def post_step(self, group: sim.Group):
        itemss = [sim.shape_query(group.world, self.shape, body.transform)
                  for body in group.bodies]
        inventory = group.get(Inventory)[0]
        for body, items in zip(group.bodies, itemss):
            [inventory.pickup(body, item) for item in items]


# TODO make it a discrete action which consumes the N-th-last item instead of
# the last
# provides an use action which consumes the last item in each inventory of the
# body
class UseLast(sim.Module):
    drift: bool
    uses: List[bool]

    def __init__(self, drift: bool = False):
        self.drift = drift

    def post_reset(self, group: sim.Group):
        self.uses = []

    def pre_step(self, group: sim.Group):
        missing = len(group.bodies) - len(self.uses)
        if missing > 0:
            self.uses += [False] * missing
        inventories = group.get(Inventory)
        for user, use in zip(group.bodies, self.uses):
            if use:
                [inventory.use(user) for inventory in inventories]
        if not self.drift:
            self.uses = []


class GiveLast(sim.Module):
    shape: b2Shape
    drift: bool
    give: List[bool]
    takers: List[Optional[b2Body]]
    # if a taker is in strangers[giver], the give action fails
    strangers: Dict[b2Body, Set[b2Body]]

    def __init__(self, shape: b2Shape, drift: bool = False):
        self.shape = shape
        self.drift = drift

    def add_strangers(self, body: b2Body, strangers):
        if body not in self.strangers:
            self.strangers[body] = set()
        if not isinstance(strangers, set):
            strangers = set(strangers)
        self.strangers[body] |= strangers

    def post_reset(self, group: sim.Group):
        self.give = []
        self.strangers = {}

    def pre_step(self, group: sim.Group):
        self._update_takers(group)
        missing = len(group.bodies) - len(self.give)
        if missing > 0:
            self.give += [False] * missing
        inventories = group.get(Inventory)
        for giver, taker, gives in zip(group.bodies, self.takers, self.give):
            if not gives or taker is None:
                continue
            if giver in self.strangers and taker in self.strangers[giver]:
                #                 indexed_bodies = group.get(sim.IndexBodies)[0].bodies
                #                 giver_id = indexed_bodies.index(giver)
                #                 taker_id = indexed_bodies.index(taker)
                #                 print(f'give action failed: g = {giver_id}, taker = {taker_id}, strangers = {stranger_ids}')
                continue
            [inventory.give(giver, taker) for inventory in inventories]
        if not self.drift:
            self.give = []

    def _update_takers(self, group: sim.Group):
        shape = self.shape
        neighbourss = [sim.shape_query(group.world, shape, body.transform)
                       for body in group.bodies]
        self.takers = []
        for body, neighbours in zip(group.bodies, neighbourss):
            min_distance: float = float('inf')
            taker: Optional[b2Body] = None
            for neighbour in neighbours:
                # TODO also check that the taker is in the right group
                if neighbour == body:
                    continue
                distance: float = (body.position - neighbour.position).length
                if distance < min_distance:
                    min_distance = distance
                    taker = neighbour
            self.takers.append(taker)


# drop items on death, scattering them randomly around the dead body; should
# be placed before the inventory module so that it can drop the items before
# the inventory vanishes
class DeathDrop(sim.Module):
    radius: float  # radius of the scatter (fixed for now)
    rng: np.random.Generator
    group: sim.Group

    def __init__(self, radius: float):
        self.radius = radius

    def post_reset(self, group: sim.Group):
        self.group = group

    def pre_despawn(self, bodies: List[b2Body]):
        for m in self.group.get(Inventory):
            fulls = [m.full(body) for body in bodies]
            n_samples = np.sum(fulls, dtype=int)
            all_angles = list(2 * np.pi * self.rng.random(n_samples))
            for body, full in zip(bodies, fulls):
                angles = [all_angles.pop() for _ in range(full)]
                offsets = [sim.from_polar(self.radius, angle)
                           for angle in angles]
                m.drop_all(body, offsets)


# health, attacks, healing

# gives health to all bodies in its group and kills them when their health <=
# 0
class Health(sim.Module):
    # starting health for all bodies with this module
    health: int
    # can be set by others to disable damage and heal entirely (overrides behaviour of vulnerabilities)
    immune: bool
    # if an entry for a body is present, it can only be damaged by causes in that set; if the entry is not present, it can be damaged by anyone; if the entry has 0 elements, the body is immune
    vulnerabilities: Dict[b2Body, Set[Any]]
    # opposite of vulnerabilities; they are checked first, so if a cause appears in both, immunities take precedence
    immunities: Dict[b2Body, Set[Any]]
    healths: Dict[b2Body, int]
    causes: Dict[b2Body, Any]
    on_death: Optional[Callable]

    def __init__(self, health: int, on_death: Optional[Callable] = None):
        self.health = health
        self.on_death = on_death

    def post_reset(self, group: sim.Group):
        # TODO do this in post_spawn if attr is not set yet
        self.healths = {body: self.health for body in group.bodies}
        self.causes = {}
        self.vulnerabilities = {}
        self.immunities = {}
        self.immune = False

    def post_step(self, group: sim.Group):
        # TODO do this in post_spawn
        for body in group.bodies:
            if body not in self.healths:
                self.healths[body] = self.health
        dead = [body for body, health in self.healths.items() if health <= 0]
        group.despawn(dead)

    def pre_despawn(self, bodies: List[b2Body]):
        for body in bodies:
            cause = self.causes.get(body)
            if self.on_death is not None:
                self.on_death(body, cause)
            del self.healths[body]
            if body in self.causes:
                del self.causes[body]
            if body in self.vulnerabilities:
                del self.vulnerabilities[body]
            if body in self.immunities:
                del self.immunities[body]

    # convenience method to add vulnerabilities
    def add_vulnerabilities(self, body: b2Body, causes):
        if body not in self.vulnerabilities:
            self.vulnerabilities[body] = set()
        if not isinstance(causes, set):
            causes = set(causes)
        self.vulnerabilities[body] |= causes

    # removes given vulnerabilities; if the resulting list is empty and del_empty is True, the body entry gets removed from the vulnerabilities dict
    def remove_vulnerabilities(
            self, body: b2Body, causes, del_empty: bool = True
    ):
        if body not in self.vulnerabilities:
            return
        if not isinstance(causes, set):
            causes = set(causes)
        self.vulnerabilities[body] -= causes
        if del_empty and len(self.vulnerabilities[body]) == 0:
            del self.vulnerabilities[body]

    # convenience method to add immunities
    def add_immunities(self, body: b2Body, causes):
        if body not in self.immunities:
            self.immunities[body] = set()
        if not isinstance(causes, set):
            causes = set(causes)
        self.immunities[body] |= causes

    # removes given immunities; if the resulting list is empty and del_empty is True, the body entry gets removed from the immunities dict
    def remove_immunities(
            self, body: b2Body, causes, del_empty: bool = True
    ):
        if body not in self.immunities:
            return
        if not isinstance(causes, set):
            causes = set(causes)
        self.immunities[body] -= causes
        if del_empty and len(self.immunities[body]) == 0:
            del self.immunities[body]

    def _change_health(self, body: b2Body, delta: int, cause: Any = None):
        if body not in self.healths:
            return
        V = self.vulnerabilities.get(body)
        I = self.immunities.get(body)
        if I is not None and cause in I:
            return
        if V is not None and cause not in V:
            return
        self.healths[body] += delta
        self.causes[body] = cause

    def damage(self, body: b2Body, damage: int, cause: Any = None):
        self._change_health(body, -damage, cause)

    def heal(self, body: b2Body, healing: int, cause: Any = None):
        self._change_health(body, healing, cause)


# gives an short-range attack that damages the target (if it has an health
# module)
class ContinuousMelee(sim.Module):
    range: float
    damage: int
    drift: bool
    get_attacker: Optional[Callable]
    # TODO init these in the post reset
    targets: List[Optional[b2Body]] = []
    origins: List[b2Vec2] = []
    endpoints: List[b2Vec2] = []
    attacks: List[bool] = []

    def __init__(
            self, range: float, damage: int, drift: bool = False,
            get_attacker: Optional[Callable] = None
    ):
        self.range = range
        self.damage = damage
        self.drift = drift
        self.get_attacker = get_attacker

    def pre_step(self, group: sim.Group):
        self.origins = [body.position for body in group.bodies]
        hands = [sim.from_polar(length=self.range, angle=body.angle)
                 for body in group.bodies]
        self.endpoints = [a + d for a, d in zip(self.origins, hands)]
        scans = [sim.laser_scan(group.world, a, b)
                 for a, b in zip(self.origins, self.endpoints)]
        self.targets = [scan and scan[0].body for scan in scans]
        missing = len(group.bodies) - len(self.attacks)
        if missing > 0:
            self.attacks += [False] * missing
        for body, target, attack \
                in zip(group.bodies, self.targets, self.attacks):
            if target is not None and attack:
                self._attack(target, self.damage, attacker=body)
        if not self.drift:
            self.attacks = []

    def _attack(self, target: b2Body, damage: int, attacker: b2Body):
        healths = sim.Group.body_group(target).get(Health)
        if self.get_attacker is not None:
            attacker = self.get_attacker(attacker)
        for health in healths:
            health.damage(target, damage, cause=attacker)  # type: ignore


# gives a "strong" short-range attack with cooldown that damages the target (if it has an health module)
class Melee(sim.Module):
    range: float
    damage: int
    cooldown: int
    drift: bool
    # which attribute of the attacker body is used as the cause of damage
    get_attacker: Optional[Callable]
    # TODO init these in the post reset
    targets: List[Optional[b2Body]] = []
    origins: List[b2Vec2] = []
    endpoints: List[b2Vec2] = []
    attacks: List[bool] = []

    def __init__(
            self, range: float, damage: int, cooldown: int, drift: bool = False,
            get_attacker: Optional[Callable] = None
    ):
        self.range = range
        self.damage = damage
        self.cooldown = cooldown
        self.drift = drift
        self.get_attacker = get_attacker

    def post_reset(self, group: sim.Group):
        self.cooldowns = {}

    def pre_step(self, group: sim.Group):
        self.origins = [body.position for body in group.bodies]
        hands = [sim.from_polar(length=self.range, angle=body.angle)
                 for body in group.bodies]
        self.endpoints = [a + d for a, d in zip(self.origins, hands)]
        scans = [sim.laser_scan(group.world, a, b)
                 for a, b in zip(self.origins, self.endpoints)]
        self.targets = [scan and scan[0].body for scan in scans]
        missing = len(group.bodies) - len(self.attacks)
        if missing > 0:
            self.attacks += [False] * missing
        for body, target, attack \
                in zip(group.bodies, self.targets, self.attacks):
            on_cooldown = body in self.cooldowns
            if target is not None and attack and not on_cooldown:
                self._attack(target, self.damage, attacker=body)
                self.cooldowns[body] = self.cooldown
        if not self.drift:
            self.attacks = []
        # unmark bodies that have fully cooled down
        cooleddown = set()
        for k in self.cooldowns.keys():
            self.cooldowns[k] -= 1
            assert self.cooldowns[k] >= 0
            if self.cooldowns[k] == 0:
                cooleddown.add(k)
        [self.cooldowns.pop(k) for k in cooleddown]

    def _attack(self, target: b2Body, damage: int, attacker: b2Body):
        healths = sim.Group.body_group(target).get(Health)
        if self.get_attacker is not None:
            attacker = self.get_attacker(attacker)
        for health in healths:
            health.damage(target, damage, cause=attacker)  # type: ignore


# adds deaths to a buffer (not with body objects, but with body indices) that can be flushed with flush method
class TrackKills(sim.Module):

    def post_reset(self, group: sim.Group):
        for m in group.get(Health):
            m.on_death = self
        self.kills = []

    # on_death callback for health module(s)
    def __call__(self, body: b2Body, cause: Any):
        self.kills.append((body, cause))

    def flush(self):
        kills = self.kills
        self.kills = []
        return kills


# heals the target by the specified amount of health
class Heal(Item):
    prototype: sim.Prototype
    healing: int

    def __init__(self, healing: int, prototype: sim.Prototype):
        self.healing = healing
        self.prototype = prototype

    def use(self, user: b2Body, data: Any):
        healths = sim.Group.body_group(user).get(Health)
        for health in healths:
            health.heal(user, self.healing)  # type: ignore


# disables PvP damage for a fixed amount of steps at reset
class ImmunityPhase(sim.Module):
    cooldown: int
    t: int
    finished: bool

    def __init__(self, cooldown: int):
        self.cooldown = cooldown

    def post_reset(self, group: sim.Group):
        self.health = group.get(Health)[0]
        self.health.immune = True
        self.t = self.cooldown
        self.finished = False

    def post_step(self, group: sim.Group):
        if self.finished:
            return
        if self.t > 0:
            self.t -= 1
        if self.t == 0:
            self.health.immune = False
            self.finished = True


# map features: safe zone, terrain, etc.

# spawns 4 immovable, thick walls enclosing a room of given size
class ThickRoomWalls(sim.Module):
    prototypes: List[sim.Prototype]
    placements: List[b2Vec2]

    def __init__(self, room_size: float, wall_aspect_ratio: float = 100):
        height = room_size
        width = height / wall_aspect_ratio
        shape = sim.rect_shape(width=width, height=height)
        self.prototypes = [sim.Prototype(shape=shape, dynamic=False)] * 4
        offset = room_size / 2
        self.placements = [
            (-offset, 0, 0),  # west wall
            (0, offset, np.pi / 2),  # north wall
            (offset, 0, 0),  # east wall
            (0, -offset, np.pi / 2), ]  # south wall

    def post_reset(self, group: sim.Group):
        group.spawn(self.prototypes, self.placements)


# shrinking & moving safe zones (e.g. like fortnite); all bodies in their
# group take constant damage when outside the zone; each zone in the sequence
# has 2 phases: cooldown and "shrink & move" to the next zone. The last zone
# is void, so all bodies in the group take damage
class SafeZone(sim.Module):
    phases: int
    cooldown: int
    damage: int
    radiuses: List[float]
    centers: List[b2Vec2]
    phase: int
    t_cooldown: int
    t_shrink: int
    endgame: bool
    zone: Tuple[b2CircleShape, b2Transform]
    outliers: List[b2Body]
    room_size: float
    rng: np.random.Generator

    def __init__(
            self, phases: int, cooldown: int, damage: int,
            radiuses: List[float], centers: List[sim.Vec2], room_size: Optional[float] = None):
        self.phases = phases
        self.cooldown = cooldown
        self.damage = damage
        self.radiuses = list(radiuses)
        self.radiuses.append(0)
        if centers == 'random':
            assert room_size is not None
            self.room_size = room_size
        else:
            self.centers = [b2Vec2(c) for c in centers]
            self.centers.append(b2Vec2(0, 0))

    # compute the maximum number of steps an agent can survive, provided it always stays inside the safe zone, and also considering the steps it can survive after the zone becomes void
    def max_lifespan(self, max_health: int):
        return (self.phases - 1) * 2 * self.cooldown + math.ceil(max_health / self.damage)

    def post_reset(self, group: sim.Group):
        # This means 'random' was given for 'centers' in init.
        if hasattr(self, 'room_size'):
            reverse_centers = []
            for r in reversed(self.radiuses):
                L = self.room_size - 2 * r
                cx = (self.rng.random() * L) - L / 2
                cy = (self.rng.random() * L) - L / 2
                reverse_centers.append(b2Vec2(cx, cy))
            self.centers = list(reversed(reverse_centers))
        self.t_cooldown = self.cooldown
        self.t_shrink = 0
        self.phase = 0
        self.endgame = False
        shape = sim.circle_shape(self.radiuses[0])
        transform = sim.transform(translation=self.centers[0])
        self.zone = (shape, transform)
        self.outliers = []

    def post_step(self, group: sim.Group):
        self.outliers = []
        healths = group.get(Health)
        shape, transform = self.zone
        for body in group.bodies:
            if self.endgame \
                    or not shape.TestPoint(transform, body.worldCenter):
                self.outliers.append(body)
                [health.damage(body, self.damage, cause=self)
                 for health in healths]
        self.tick()

    @property
    def shrinking(self) -> bool:
        assert (self.t_cooldown == 0) != (self.t_shrink == 0)
        return self.t_cooldown == 0

    # advances time by 1 and updates the zone and phase
    def tick(self):
        if self.shrinking:
            self._tick_shrink()
        else:
            self._tick_cooldown()

    def _tick_cooldown(self):
        self.t_cooldown -= 1
        if self.t_cooldown > 0:
            return
        self.t_shrink = self.cooldown

    def _tick_shrink(self):
        if self.endgame:
            return
        self.t_shrink -= 1
        if self.t_shrink > 0:
            self._update_shrinking_zone()
            return
        self.t_cooldown = self.cooldown
        self.phase = self.phase + 1
        shape = sim.circle_shape(self.radiuses[self.phase])
        transform = sim.transform(translation=self.centers[self.phase])
        self.zone = (shape, transform)
        if self.phase == self.phases - 1:
            self.endgame = True

    def _update_shrinking_zone(self):
        t = self.t_shrink / self.cooldown
        r1, r2 = self.radiuses[self.phase: self.phase + 2]
        c1, c2 = self.centers[self.phase: self.phase + 2]
        radius = t * r1 + (1 - t) * r2
        center = t * c1 + (1 - t) * c2
        shape = sim.circle_shape(radius)
        transform = sim.transform(translation=center)
        self.zone = (shape, transform)


# breakable/placeable objects

# items that spawn objects when used; usually used in conjunction with an Object module that refers to it (usually in another group), see below
# uses item data to store the prototype for the spawned object
class ObjectItem(Item):
    prototype: sim.Prototype
    offset: float
    # this is automatically set by the Object module that uses this module;
    # rememeber there can be only 1 for now!
    object_group: sim.Group

    def __init__(self, prototype: sim.Prototype, offset: float):
        self.prototype = prototype
        self.offset = offset

    def use(self, user: b2Body, data: Any):
        assert isinstance(data, sim.Prototype)
        offset = sim.from_polar(self.offset, user.angle)
        self.object_group.spawn(
            [data],
            [user.position + offset]
        )


# things that, instead of completely despawning, drop as items when killed;
# the dropped item will then spawn back the original object
# this requires an ObjectItem module in a group (usually different)
class Object(sim.Module):
    item_module: ObjectItem
    next_spawns: List[Tuple]  # TODO specify type better

    def __init__(self, item_module: ObjectItem):
        self.item_module = item_module

    def post_reset(self, group: sim.Group):
        self.item_module.object_group = group
        self.next_spawns = []

    def pre_step(self, group: sim.Group):
        for pos, proto in self.next_spawns:
            self.item_module.drop(placement=pos, data=proto)
        self.next_spawns = []

    def pre_despawn(self, bodies: List[b2Body]):
        self.next_spawns += list(
            [(body.position, sim.prototype(body)) for body in bodies]
        )


# a variant of object items that treats the data as a pair (proto, owner) and spawns objects that can only be damaged by owner; to be used with OwnedObject
class OwnedObjectItem(ObjectItem):
    prototype: sim.Prototype
    offset: float
    # this is automatically set by the Object module that uses this module;
    # rememeber there can be only 1 for now!
    object_group: sim.Group

    def __init__(self, prototype: sim.Prototype, offset: float):
        self.prototype = prototype
        self.offset = offset

    def use(self, user: b2Body, data: Any):
        proto, owner = data
        offset = sim.from_polar(self.offset, user.angle)
        self.object_group.spawn(
            [proto],
            [user.position + offset]
        )
        for m in self.object_group.get(Health):
            m.add_vulnerabilities(self.object_group.bodies[-1], [owner])


# a variant of objects that records the cause of death and uses it to only allows further instances of that object to be damaged only by that cause
class OwnedObject(Object):
    item_module: ObjectItem
    next_spawns: List[Tuple]  # TODO specify type better

    def __init__(self, item_module: ObjectItem):
        self.item_module = item_module

    def post_reset(self, group: sim.Group):
        self.item_module.object_group = group
        self.next_spawns = []
        self.group = group
        for m in group.get(Health):
            m.on_death = self

    def pre_step(self, group: sim.Group):
        for pos, data in self.next_spawns:
            self.item_module.drop(placement=pos, data=data)
        self.next_spawns = []

    def pre_despawn(self, bodies):
        pass

    # the on_death callback for the health module(s)
    def __call__(self, body: b2Body, cause: Any):
        self.next_spawns.append((body.position, (sim.prototype(body), cause)))


# teams

class TeamBadge:
    def __init__(self, i):
        self.i = i


# divides the bodies in the group into 2 teams; mainly used for immunity between team memebers and convenience functions; automatically changes the causes of damage for Melee and ContinuousMelee to the team badge, and adds immunities/strangers to their Health module
# should be put after index bodies and any initial spawner
class TwoTeams(sim.Module):

    def __init__(self, index_bodies: sim.IndexBodies = None):
        self.index_bodies = index_bodies

    def post_reset(self, group):
        self.team_badges = [TeamBadge(0), TeamBadge(1)]
        for m in group.get(Melee):
            m.get_attacker = lambda b: self.team_badges[self.get_team_id(b)]
        for m in group.get(ContinuousMelee):
            m.get_attacker = lambda b: self.team_badges[self.get_team_id(b)]
        if self.index_bodies is None:
            self.index_bodies = group.get(sim.IndexBodies)[0]
        # the first body id of the second team
        self.split_id = len(self.index_bodies.bodies) // 2
        self.teams = [
            list(range(0, self.split_id)),
            list(range(self.split_id, len(self.index_bodies.bodies))),
        ]
        for body in group.bodies:
            for m in group.get(Health):
                m.add_immunities(
                    body, [self.team_badges[self.get_team_id(body)]]
                )
            for m in group.get(GiveLast):
                opponent_team_id = int(not bool(self.get_team_id(body)))
                m.add_strangers(
                    body,
                    set(self.index_bodies.bodies[i]
                        for i in self.teams[opponent_team_id]),
                )
        # for i, body in enumerate(self.index_bodies.bodies):
        #    print(f'Body {i} is in team {self.get_team_id(body)}')

    def get_team_id(self, body_or_id):
        body_id = body_or_id
        if isinstance(body_or_id, b2Body):
            if body_or_id not in self.index_bodies.bodies:
                return None
            body_id = self.index_bodies.bodies.index(body_or_id)
        if body_id < self.split_id:
            return 0
        else:
            return 1

    def get_team(self, body_or_id):
        return self.teams[self.get_team_id(body_or_id)]

    def get_team_bodies(self, body_or_id):
        return [
            self.index_bodies.bodies[i]
            for i in self.get_team(body_or_id)
            if self.index_bodies.bodies[i] is not None
        ]

    def team_is_alive(self, team_id):
        return any(
            self.index_bodies.bodies[i] is not None
            for i in self.teams[team_id]
        )


# utilties

def square_grid(grid_size: int, floor_size: float) -> List[b2Vec2]:
    centers = np.arange(grid_size) / grid_size + 0.5 / grid_size
    centers = floor_size * centers - floor_size / 2.
    ii = np.arange(grid_size ** 2) % grid_size
    jj = np.arange(grid_size ** 2) // grid_size
    return [b2Vec2(centers[i], centers[j]) for i, j in zip(ii, jj)]


