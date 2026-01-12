"""Base game objects."""

from typing import Optional, List


class FastObject:
    def __init__(self, name: str):
        self.name = name
        self.is_container = False
        self.is_open = False
        self.is_openable = False
        self.is_movable = False
        self.current_container: Optional['FastObject'] = None
        self.contents: List['FastObject'] = []
        self.indefinite = "an" if name[0].lower() in "aeiou" else "a"
    
    def add_object(self, obj: 'FastObject'):
        if obj.current_container is not None:
            obj.remove_from_current_container()
        self.contents.append(obj)
        obj.current_container = self
    
    def remove_object(self, obj: 'FastObject'):
        if obj in self.contents:
            self.contents.remove(obj)
            obj.current_container = None
    
    def remove_from_current_container(self):
        if self.current_container is not None:
            self.current_container.remove_object(self)
    
    def collect_visible_objects(self, objs_in: List['FastObject'] = None) -> List['FastObject']:
        if objs_in is None:
            objs_in = []
        
        for obj in self.contents:
            objs_in.append(obj)
            if obj.is_container and obj.is_open:
                obj.collect_visible_objects(objs_in)
        
        return objs_in
    
    def get_description(self) -> str:
        return f"{self.indefinite} {self.name}"


class Door(FastObject):
    def __init__(self, door_description: str, is_open: bool = False):
        super().__init__("door")
        self.door_description = door_description
        self.is_open = is_open
        self.is_movable = False
    
    def get_description(self) -> str:
        return self.door_description


class Container(FastObject):
    def __init__(self, name: str, is_open: bool = False, is_openable: bool = True):
        super().__init__(name)
        self.is_container = True
        self.is_open = is_open
        self.is_openable = is_openable
    
    def get_description(self) -> str:
        if not self.is_open:
            return f"{self.indefinite} {self.name} that is closed"
        else:
            if not self.contents:
                return f"an open {self.name}, that is empty"
            else:
                items = ", ".join([obj.get_description() for obj in self.contents])
                return f"an open {self.name}, that contains {items}"


class Surface(FastObject):
    def __init__(self, name: str):
        super().__init__(name)
        self.is_container = True
        self.is_open = True
        self.is_openable = False
    
    def get_description(self) -> str:
        if not self.contents:
            return f"{self.indefinite} {self.name}, that has nothing on it"
        else:
            items = ", ".join([obj.get_description() for obj in self.contents])
            return f"{self.indefinite} {self.name} that has {items} on it"


class Coin(FastObject):
    def __init__(self):
        super().__init__("coin")
        self.is_movable = True


class Room(FastObject):
    def __init__(self, name: str, prefers_connecting_to: List[str] = None):
        super().__init__(name)
        self.is_location = True
        self.prefers_connecting_to = prefers_connecting_to or []
        self.location_north: Optional['Room'] = None
        self.location_south: Optional['Room'] = None
        self.location_east: Optional['Room'] = None
        self.location_west: Optional['Room'] = None
        self.door_north: Optional[Door] = None
        self.door_south: Optional[Door] = None
        self.door_east: Optional[Door] = None
        self.door_west: Optional[Door] = None
        
        self.starts = [
            "In one part of the room you see ",
            "There is also ",
            "You also see ",
            "In another part of the room you see "
        ]
    
    def _mk_direction_description(self, location: Optional['Room'], door: Optional[Door], direction_name: str) -> str:
        if location is None:
            return ""
        
        if door is None:
            return f"To the {direction_name} you see the {location.name}. "
        else:
            if door.is_open:
                return f"Through an open {door.get_description()}, to the {direction_name} you see the {location.name}. "
            else:
                return f"To the {direction_name} you see a closed {door.get_description()}. "
    
    def get_description(self) -> str:
        os = []
        os.append(f"You are in the {self.name}. ")
        
        obj_descriptions = [obj.get_description() for obj in self.contents]
        for i, desc in enumerate(obj_descriptions):
            os.append(self.starts[i % len(self.starts)] + desc + ". ")
        
        os.append("\n")
        
        os.append(self._mk_direction_description(self.location_north, self.door_north, "North"))
        os.append(self._mk_direction_description(self.location_south, self.door_south, "South"))
        os.append(self._mk_direction_description(self.location_east, self.door_east, "East"))
        os.append(self._mk_direction_description(self.location_west, self.door_west, "West"))
        
        return "".join(os)

