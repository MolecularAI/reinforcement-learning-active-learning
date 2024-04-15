"""Dataclasses that describe expected AiZynth Scoring Component parameters.

Dataclasses give better code completions and code checks than plain dicts.
Combined with data validation library like apischema or pydantic (v2),
dataclasses can help validating the input.
"""

import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List, Optional, Any, Dict


@dataclass
class StockFile:
    name: str
    uri: str


@dataclass
class StockScore:
    name: str
    score: float


@dataclass
class StockDef:
    """Stock definition.

    This is alternative stock definition.
    AiZynth uses stock's names as keys in a dict,
    which makes it difficult to create a static form.

    This stock definition uses fixed keys "name", "uri", and "score",
    which is easier to put in a form.

    This stock definition can be converted to AiZynth with:
        stockfiles = {}
        for stock in params.stock.files:
            stockfiles[stock.name] = stock.uri
    """

    files: List[StockFile] = field(default_factory=list)
    scores: List[StockScore] = field(default_factory=list)


class StockProfile(str, Enum):
    IsacErm = "ISAC+ERM"
    IsacErmAcd = "ISAC+ERM+ACD"


class ReactionsProfile(str, Enum):
    ILAB = "ILAB"
    AllReactions = "AllReactions"


@dataclass
class AiZynthParams:
    number_of_steps: int = 5
    base_aizynthfinder_config: Union[None, str, pathlib.Path] = None
    custom_aizynth_command: Optional[str] = None
    time_limit_seconds: int = 120
    stock: Optional[Dict[str, Any]] = None
    scorer: Optional[Dict[str, Any]] = None
    transformation: Optional[Any] = None
    stock_profile: Optional[StockProfile] = None
    reactions_profile: Optional[ReactionsProfile] = None
