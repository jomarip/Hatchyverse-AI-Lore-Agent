from enum import Enum
from typing import Optional

class Element(Enum):
    VOID = "Void"
    PLANT = "Plant"
    WATER = "Water"
    FIRE = "Fire"
    DARK = "Dark"
    LIGHT = "Light"

class Monster:
    def __init__(
        self,
        name: str,
        element: Element,
        description: str,
        monster_id: int,
        height: int,
        weight: int,
        image: Optional[str] = None,
        sound: Optional[str] = None
    ):
        self.name = name
        self.element = element
        self.description = description
        self.monster_id = monster_id
        self.height = height
        self.weight = weight
        self.image = image
        self.sound = sound

    def __str__(self) -> str:
        return f"{self.name} ({self.element.value}) - ID: {self.monster_id}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def asset_path(self) -> str:
        """Returns the path to the monster's image asset"""
        return f"assets/monsters/{self.image}" if self.image else ""

    @property
    def sound_path(self) -> str:
        """Returns the path to the monster's sound asset"""
        return f"assets/sounds/{self.sound}" if self.sound else "" 